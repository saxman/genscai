"""Modular compartmental disease-modeling engine.

A single coupled SEIR system is integrated each step with ``scipy.integrate.solve_ivp``. The
only structural choice exposed to a caller (or an agent) is *which mechanism modules to
include* - the mixing structure and an optional host-vector module - so results stay
epidemiologically consistent regardless of which modules are active.

Mechanism modules
-----------------
- ``mass_action``: well-mixed, frequency-dependent force of infection ``beta * I / N``.
- ``metapopulation``: independent patches coupled by diffusive migration of living
  individuals toward the cross-patch mean at ``migration_rate`` per day.
- ``network``: heterogeneous-contact mean field. Transmission is scaled by
  ``mean_contacts / REFERENCE_CONTACTS``; at ``mean_contacts == REFERENCE_CONTACTS`` (10/day)
  it reduces to ``mass_action``. Denser networks spread faster.
- host-vector (optional): Ross-Macdonald-style coupling. When enabled, host-to-host
  transmission is disabled and all transmission flows through vectors, governed by
  ``biting_rate`` (not ``r0``).

Interventions are time-windowed parameter modifiers; see :class:`Intervention`.

Per-patch state carries the four SEIR compartments plus two non-physical trackers: ``C``
(cumulative infections, the integral of the S->E flux) and ``V`` (cumulative vaccinated).
The trackers let cumulative incidence be measured separately from vaccination, which also
drains ``S``. Optional vector compartments ``Sv``/``Iv`` are appended per patch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

REFERENCE_CONTACTS = 10.0
DEFAULT_VECTOR_MORTALITY = 0.1
DEFAULT_VACCINATION_COVERAGE_CAP = 0.8

INTERVENTION_KINDS = ("vaccination", "contact_reduction", "isolation", "treatment")


@dataclass
class Intervention:
    """A time-windowed modifier of the disease dynamics.

    ``magnitude`` is interpreted per ``kind``:

    - ``vaccination``: fraction of remaining susceptibles moved S->R per day (capped by the
      coverage ceiling).
    - ``contact_reduction``: fractional reduction of the transmission rate (0.5 halves beta).
    - ``isolation``: fractional reduction of the infectious compartment's contribution to the
      force of infection.
    - ``treatment``: fractional increase of the recovery rate (shortens the infectious period).

    ``end_day`` of ``None`` means the intervention stays active for the rest of the run.
    """

    kind: str
    magnitude: float
    start_day: float
    end_day: Optional[float] = None

    def is_active(self, day: float) -> bool:
        return self.start_day <= day and (self.end_day is None or day < self.end_day)


@dataclass
class ModelConfig:
    mixing: str
    population: float
    initial_infected: float
    beta: float
    sigma: float
    gamma: float
    num_patches: int = 1
    migration_rate: float = 0.0
    mean_contacts: float = REFERENCE_CONTACTS
    include_vector: bool = False
    biting_rate: float = 0.0
    vector_population: float = 0.0
    vector_mortality: float = DEFAULT_VECTOR_MORTALITY
    coverage_cap: float = DEFAULT_VACCINATION_COVERAGE_CAP


# Per-patch compartment layout. Trackers C and V are non-physical accumulators.
_HOST_COMPARTMENTS = ("S", "E", "I", "R", "C", "V")
_VECTOR_COMPARTMENTS = ("Sv", "Iv")


class Simulation:
    """A stateful disease-modeling session: assemble a model, advance it, intervene, rewind."""

    def __init__(self) -> None:
        self.config: Optional[ModelConfig] = None
        self.interventions: list[Intervention] = []
        self._compartments: tuple[str, ...] = ()
        self._index: dict[str, int] = {}
        self._patch_population: float = 0.0
        self._vector_patch_population: float = 0.0
        self._initial_state: Optional[np.ndarray] = None
        self._times: list[float] = []
        self._states: list[np.ndarray] = []
        self._checkpoints: dict[str, dict] = {}

    # -- construction --------------------------------------------------------------------

    def assemble(
        self,
        *,
        mixing: str = "mass_action",
        population: float,
        initial_infected: float,
        r0: float,
        incubation_days: float,
        infectious_days: float,
        num_patches: int = 1,
        migration_rate: float = 0.0,
        mean_contacts: float = REFERENCE_CONTACTS,
        include_vector: bool = False,
        biting_rate: float = 0.0,
        vector_population: float = 0.0,
    ) -> "Simulation":
        if mixing not in ("mass_action", "metapopulation", "network"):
            raise ValueError(f"unknown mixing '{mixing}'")
        if num_patches < 1:
            raise ValueError("num_patches must be >= 1")

        gamma = 1.0 / infectious_days
        sigma = 1.0 / incubation_days
        beta = r0 * gamma

        if vector_population <= 0:
            vector_population = population

        self.config = ModelConfig(
            mixing=mixing,
            population=population,
            initial_infected=initial_infected,
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            num_patches=num_patches,
            migration_rate=migration_rate,
            mean_contacts=mean_contacts,
            include_vector=include_vector,
            biting_rate=biting_rate,
            vector_population=vector_population,
        )
        self.interventions = []
        self._checkpoints = {}

        self._compartments = _HOST_COMPARTMENTS + (_VECTOR_COMPARTMENTS if include_vector else ())
        self._index = {name: i for i, name in enumerate(self._compartments)}
        self._patch_population = population / num_patches
        self._vector_patch_population = vector_population / num_patches

        self._initial_state = self._build_initial_state()
        self._times = [0.0]
        self._states = [self._initial_state.copy()]
        return self

    def _build_initial_state(self) -> np.ndarray:
        state = np.zeros((self.config.num_patches, len(self._compartments)))
        state[:, self._index["S"]] = self._patch_population
        # Seed the outbreak in patch 0 only.
        state[0, self._index["I"]] = self.config.initial_infected
        state[0, self._index["S"]] -= self.config.initial_infected
        if self.config.include_vector:
            state[:, self._index["Sv"]] = self._vector_patch_population
        return state

    # -- interventions -------------------------------------------------------------------

    def apply_intervention(self, kind: str, magnitude: float, start_day: float, end_day: Optional[float] = None) -> None:
        if kind not in INTERVENTION_KINDS:
            raise ValueError(f"unknown intervention kind '{kind}'; expected one of {INTERVENTION_KINDS}")
        self.interventions.append(Intervention(kind, magnitude, start_day, end_day))

    def list_interventions(self) -> list[Intervention]:
        return list(self.interventions)

    def remove_intervention(self, index: int) -> Intervention:
        return self.interventions.pop(index)

    def _active_modifiers(self, day: float) -> dict:
        contact_factor = 1.0
        isolation_factor = 1.0
        treatment_factor = 1.0
        vaccination_rate = 0.0
        for iv in self.interventions:
            if not iv.is_active(day):
                continue
            if iv.kind == "contact_reduction":
                contact_factor *= 1.0 - iv.magnitude
            elif iv.kind == "isolation":
                isolation_factor *= 1.0 - iv.magnitude
            elif iv.kind == "treatment":
                treatment_factor *= 1.0 + iv.magnitude
            elif iv.kind == "vaccination":
                vaccination_rate += iv.magnitude
        return {
            "contact_factor": contact_factor,
            "isolation_factor": isolation_factor,
            "treatment_factor": treatment_factor,
            "vaccination_rate": vaccination_rate,
        }

    # -- dynamics ------------------------------------------------------------------------

    def _mixing_factor(self) -> float:
        if self.config.mixing == "network":
            return max(self.config.mean_contacts, 0.0) / REFERENCE_CONTACTS
        return 1.0

    def _derivatives(self, day: float, flat_state: np.ndarray) -> np.ndarray:
        cfg = self.config
        idx = self._index
        n_comp = len(self._compartments)
        state = flat_state.reshape(cfg.num_patches, n_comp)
        derivs = np.zeros_like(state)

        mods = self._active_modifiers(day)
        beta_eff = cfg.beta * mods["contact_factor"] * self._mixing_factor()
        gamma_eff = cfg.gamma * mods["treatment_factor"]

        susceptible = state[:, idx["S"]]
        exposed = state[:, idx["E"]]
        infectious = state[:, idx["I"]]
        vaccinated = state[:, idx["V"]]
        host_n = susceptible + exposed + infectious + state[:, idx["R"]]
        safe_n = np.where(host_n > 0, host_n, 1.0)

        if cfg.include_vector:
            # Vector-borne: host-to-host transmission disabled, all transmission via vectors.
            susceptible_vectors = state[:, idx["Sv"]]
            infectious_vectors = state[:, idx["Iv"]]
            vector_n = susceptible_vectors + infectious_vectors
            safe_vector_n = np.where(vector_n > 0, vector_n, 1.0)
            force_on_hosts = cfg.biting_rate * (infectious_vectors / safe_vector_n)
            force_on_vectors = cfg.biting_rate * (infectious / safe_n)
            new_vector_infections = force_on_vectors * susceptible_vectors
            mu = cfg.vector_mortality
            derivs[:, idx["Sv"]] = mu * vector_n - new_vector_infections - mu * susceptible_vectors
            derivs[:, idx["Iv"]] = new_vector_infections - mu * infectious_vectors
        else:
            force_on_hosts = beta_eff * (infectious * mods["isolation_factor"]) / safe_n

        new_infections = force_on_hosts * susceptible

        cap = cfg.coverage_cap * self._patch_population
        vaccination_flow = np.where(vaccinated < cap, mods["vaccination_rate"] * susceptible, 0.0)

        derivs[:, idx["S"]] += -new_infections - vaccination_flow
        derivs[:, idx["E"]] += new_infections - cfg.sigma * exposed
        derivs[:, idx["I"]] += cfg.sigma * exposed - gamma_eff * infectious
        derivs[:, idx["R"]] += gamma_eff * infectious + vaccination_flow
        derivs[:, idx["C"]] += new_infections
        derivs[:, idx["V"]] += vaccination_flow

        if cfg.mixing == "metapopulation" and cfg.num_patches > 1 and cfg.migration_rate > 0:
            for name in ("S", "E", "I", "R"):
                column = state[:, idx[name]]
                derivs[:, idx[name]] += cfg.migration_rate * (column.mean() - column)

        return derivs.reshape(-1)

    # -- integration ---------------------------------------------------------------------

    def advance(self, days: float) -> "Simulation":
        if self.config is None:
            raise RuntimeError("call assemble() before advance()")
        start = self._times[-1]
        stop = start + days
        steps = max(int(round(days)), 1)
        eval_times = np.linspace(start, stop, steps + 1)

        solution = solve_ivp(
            self._derivatives,
            (start, stop),
            self._states[-1].reshape(-1),
            t_eval=eval_times,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        if not solution.success:
            raise RuntimeError(f"integration failed: {solution.message}")

        n_comp = len(self._compartments)
        for col in range(1, solution.y.shape[1]):
            self._times.append(float(solution.t[col]))
            self._states.append(solution.y[:, col].reshape(self.config.num_patches, n_comp))
        return self

    # -- state and metrics ---------------------------------------------------------------

    @property
    def day(self) -> float:
        return self._times[-1]

    def _current(self) -> np.ndarray:
        return self._states[-1]

    def state(self) -> dict:
        idx = self._index
        current = self._current()
        result = {
            "susceptible": float(current[:, idx["S"]].sum()),
            "exposed": float(current[:, idx["E"]].sum()),
            "infectious": float(current[:, idx["I"]].sum()),
            "recovered": float(current[:, idx["R"]].sum()),
        }
        if self.config.include_vector:
            result["vector_infectious"] = float(current[:, idx["Iv"]].sum())
        return result

    def infectious_by_patch(self) -> np.ndarray:
        return self._current()[:, self._index["I"]].copy()

    def cumulative_infections_by_patch(self) -> np.ndarray:
        return self._current()[:, self._index["C"]].copy()

    def metrics(self) -> dict:
        idx = self._index
        infectious_series = np.array([s[:, idx["I"]].sum() for s in self._states])
        peak_position = int(infectious_series.argmax())
        cumulative_infections = float(self._current()[:, idx["C"]].sum())

        metrics = dict(self.state())
        metrics.update(
            day=self.day,
            peak_infectious=float(infectious_series[peak_position]),
            peak_day=float(self._times[peak_position]),
            cumulative_infections=cumulative_infections,
            attack_rate=cumulative_infections / self.config.population,
            r_effective=self._effective_reproduction_number(),
        )
        return metrics

    def _effective_reproduction_number(self) -> Optional[float]:
        if self.config.include_vector:
            return None
        mods = self._active_modifiers(self.day)
        beta_eff = self.config.beta * mods["contact_factor"] * self._mixing_factor() * mods["isolation_factor"]
        gamma_eff = self.config.gamma * mods["treatment_factor"]
        susceptible = self._current()[:, self._index["S"]].sum()
        return float((beta_eff / gamma_eff) * (susceptible / self.config.population))

    # -- checkpoint / rewind / reset -----------------------------------------------------

    def checkpoint(self, label: str) -> None:
        self._checkpoints[label] = {
            "times": list(self._times),
            "states": [s.copy() for s in self._states],
            "interventions": copy.deepcopy(self.interventions),
        }

    def rewind(self, label: str) -> None:
        if label not in self._checkpoints:
            raise KeyError(f"no checkpoint named '{label}'")
        snapshot = self._checkpoints[label]
        self._times = list(snapshot["times"])
        self._states = [s.copy() for s in snapshot["states"]]
        self.interventions = copy.deepcopy(snapshot["interventions"])

    def reset(self) -> "Simulation":
        """Return to day 0 keeping the assembled model; clear interventions to try another."""
        if self.config is None:
            raise RuntimeError("call assemble() before reset()")
        self._times = [0.0]
        self._states = [self._initial_state.copy()]
        self.interventions = []
        return self
