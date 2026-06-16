"""Modular compartmental disease-modeling engine.

A single coupled SEIR(+hospitalization) system is integrated each step with
``scipy.integrate.solve_ivp``. The only structural choice exposed to a caller (or an agent)
is *which mechanism modules to include* - the mixing structure and an optional host-vector
module - so results stay epidemiologically consistent regardless of which modules are active.

This is a teaching engine, not a validated public-health model. Its outputs are illustrative.
The rigorous-analysis methods below (:meth:`Simulation.calibrate_r0`,
:meth:`Simulation.optimize_intervention`, :meth:`Simulation.run_ensemble`,
:meth:`Simulation.sensitivity_analysis`) exist so that any conclusion drawn from the model is
grounded in data and carries an uncertainty range, rather than being a single point estimate
read off one deterministic run.

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

Disease burden
--------------
Infectious hosts recover or are hospitalized (``hospitalized_fraction``); hospitalized hosts
recover or die. Mortality rises when hospital occupancy exceeds ``hospital_capacity`` (the
crisis-care effect, scaled by ``crisis_mortality_multiplier``), which is why keeping peak
hospital demand under capacity is the outcome that matters, not infections alone.

Per-patch state carries the SEIR compartments plus the current hospitalized count ``H`` and
three non-physical trackers: ``D`` (cumulative deaths), ``C`` (cumulative infections, the
integral of the S->E flux) and ``V`` (cumulative vaccinated). The trackers let cumulative
incidence, deaths, and vaccination be measured separately. Optional vector compartments
``Sv``/``Iv`` are appended per patch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from scipy.stats.qmc import LatinHypercube

REFERENCE_CONTACTS = 10.0
DEFAULT_VECTOR_MORTALITY = 0.1
DEFAULT_VACCINATION_COVERAGE_CAP = 0.8
# Acute-care beds per capita used when a caller does not specify hospital_capacity.
DEFAULT_BEDS_PER_CAPITA = 0.0025

INTERVENTION_KINDS = ("vaccination", "contact_reduction", "isolation", "treatment")

# Epidemiological parameters carrying real-world uncertainty, perturbed by the ensemble.
_UNCERTAIN_PARAMETERS = ("incubation_days", "infectious_days", "hospitalized_fraction", "hospital_fatality_fraction")

# Metrics collected per ensemble sample (all scalar).
_ENSEMBLE_METRICS = ("peak_infectious", "peak_hospitalized", "cumulative_deaths", "attack_rate", "days_over_capacity")


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
    hospitalized_fraction: float = 0.05
    hospital_stay_days: float = 10.0
    hospital_fatality_fraction: float = 0.15
    hospital_capacity: float = 0.0
    crisis_mortality_multiplier: float = 2.0


# Per-patch compartment layout. H is current hospitalizations; D/C/V are accumulators.
_HOST_COMPARTMENTS = ("S", "E", "I", "R", "H", "D", "C", "V")
_VECTOR_COMPARTMENTS = ("Sv", "Iv")


class Simulation:
    """A stateful disease-modeling session: assemble a model, advance it, intervene, rewind."""

    def __init__(self) -> None:
        self.config: Optional[ModelConfig] = None
        self.interventions: list[Intervention] = []
        self._assemble_kwargs: dict = {}
        self._compartments: tuple[str, ...] = ()
        self._index: dict[str, int] = {}
        self._patch_population: float = 0.0
        self._patch_capacity: float = 0.0
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
        hospitalized_fraction: float = 0.05,
        hospital_stay_days: float = 10.0,
        hospital_fatality_fraction: float = 0.15,
        hospital_capacity: float = 0.0,
        crisis_mortality_multiplier: float = 2.0,
    ) -> "Simulation":
        if mixing not in ("mass_action", "metapopulation", "network"):
            raise ValueError(f"unknown mixing '{mixing}'")
        if num_patches < 1:
            raise ValueError("num_patches must be >= 1")

        # Capture the exact arguments so clones (calibration, optimization, ensemble) can be
        # rebuilt by re-calling assemble with selective overrides.
        self._assemble_kwargs = dict(
            mixing=mixing,
            population=population,
            initial_infected=initial_infected,
            r0=r0,
            incubation_days=incubation_days,
            infectious_days=infectious_days,
            num_patches=num_patches,
            migration_rate=migration_rate,
            mean_contacts=mean_contacts,
            include_vector=include_vector,
            biting_rate=biting_rate,
            vector_population=vector_population,
            hospitalized_fraction=hospitalized_fraction,
            hospital_stay_days=hospital_stay_days,
            hospital_fatality_fraction=hospital_fatality_fraction,
            hospital_capacity=hospital_capacity,
            crisis_mortality_multiplier=crisis_mortality_multiplier,
        )

        gamma = 1.0 / infectious_days
        sigma = 1.0 / incubation_days
        beta = r0 * gamma

        if vector_population <= 0:
            vector_population = population
        if hospital_capacity <= 0:
            hospital_capacity = population * DEFAULT_BEDS_PER_CAPITA

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
            hospitalized_fraction=hospitalized_fraction,
            hospital_stay_days=hospital_stay_days,
            hospital_fatality_fraction=hospital_fatality_fraction,
            hospital_capacity=hospital_capacity,
            crisis_mortality_multiplier=crisis_mortality_multiplier,
        )
        self.interventions = []
        self._checkpoints = {}

        self._compartments = _HOST_COMPARTMENTS + (_VECTOR_COMPARTMENTS if include_vector else ())
        self._index = {name: i for i, name in enumerate(self._compartments)}
        self._patch_population = population / num_patches
        self._patch_capacity = hospital_capacity / num_patches
        self._vector_patch_population = vector_population / num_patches

        self._initial_state = self._build_initial_state()
        self._times = [0.0]
        self._states = [self._initial_state.copy()]
        return self

    def reconfigure(self, **overrides) -> "Simulation":
        """Re-assemble the model with selected parameters changed.

        Merges ``overrides`` into the original assemble arguments and rebuilds, returning to
        day 0 and clearing interventions. Used to adopt a calibrated parameter value.
        """
        if not self._assemble_kwargs:
            raise RuntimeError("call assemble() before reconfigure()")
        kwargs = dict(self._assemble_kwargs)
        kwargs.update(overrides)
        return self.assemble(**kwargs)

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

    def apply_intervention(
        self, kind: str, magnitude: float, start_day: float, end_day: Optional[float] = None
    ) -> None:
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
        hospitalized = state[:, idx["H"]]
        vaccinated = state[:, idx["V"]]
        host_n = susceptible + exposed + infectious + state[:, idx["R"]] + hospitalized
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

        # Infectious hosts leave at gamma_eff; a fraction are hospitalized, the rest recover.
        leaving_infectious = gamma_eff * infectious
        to_hospital = cfg.hospitalized_fraction * leaving_infectious
        recover_from_infectious = leaving_infectious - to_hospital

        # Hospitalized hosts leave at 1/stay; fatality rises with over-capacity occupancy.
        leaving_hospital = hospitalized / cfg.hospital_stay_days
        safe_h = np.where(hospitalized > 0, hospitalized, 1.0)
        over_capacity_fraction = np.where(
            hospitalized > self._patch_capacity, (hospitalized - self._patch_capacity) / safe_h, 0.0
        )
        effective_fatality = np.minimum(
            cfg.hospital_fatality_fraction * (1.0 + (cfg.crisis_mortality_multiplier - 1.0) * over_capacity_fraction),
            1.0,
        )
        deaths = leaving_hospital * effective_fatality
        recover_from_hospital = leaving_hospital - deaths

        derivs[:, idx["S"]] += -new_infections - vaccination_flow
        derivs[:, idx["E"]] += new_infections - cfg.sigma * exposed
        derivs[:, idx["I"]] += cfg.sigma * exposed - leaving_infectious
        derivs[:, idx["R"]] += recover_from_infectious + recover_from_hospital + vaccination_flow
        derivs[:, idx["H"]] += to_hospital - leaving_hospital
        derivs[:, idx["D"]] += deaths
        derivs[:, idx["C"]] += new_infections
        derivs[:, idx["V"]] += vaccination_flow

        if cfg.mixing == "metapopulation" and cfg.num_patches > 1 and cfg.migration_rate > 0:
            # Living, mobile compartments diffuse toward the cross-patch mean. Hospitalized
            # hosts and the dead do not migrate.
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
            "hospitalized": float(current[:, idx["H"]].sum()),
            "deaths": float(current[:, idx["D"]].sum()),
        }
        if self.config.include_vector:
            result["vector_infectious"] = float(current[:, idx["Iv"]].sum())
        return result

    def infectious_by_patch(self) -> np.ndarray:
        return self._current()[:, self._index["I"]].copy()

    def cumulative_infections_by_patch(self) -> np.ndarray:
        return self._current()[:, self._index["C"]].copy()

    def _series(self, compartment: str) -> np.ndarray:
        idx = self._index[compartment]
        return np.array([s[:, idx].sum() for s in self._states])

    def cumulative_infections_series_to(self, day: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, total cumulative infections) for a fresh run advanced to ``day``.

        Non-mutating: runs on a clone so the live session is unaffected. Used to generate or
        compare against observed incidence data.
        """
        clone = self._simulate(day, interventions=[])
        return np.array(clone._times), clone._series("C")

    def metrics(self) -> dict:
        idx = self._index
        infectious_series = self._series("I")
        hospital_series = self._series("H")
        peak_position = int(infectious_series.argmax())
        hospital_peak_position = int(hospital_series.argmax())
        cumulative_infections = float(self._current()[:, idx["C"]].sum())
        capacity = self.config.hospital_capacity

        metrics = dict(self.state())
        metrics.update(
            day=self.day,
            peak_infectious=float(infectious_series[peak_position]),
            peak_day=float(self._times[peak_position]),
            cumulative_infections=cumulative_infections,
            attack_rate=cumulative_infections / self.config.population,
            r_effective=self._effective_reproduction_number(),
            peak_hospitalized=float(hospital_series[hospital_peak_position]),
            peak_hospital_day=float(self._times[hospital_peak_position]),
            hospital_capacity=float(capacity),
            days_over_capacity=int((hospital_series > capacity).sum()),
            cumulative_deaths=float(self._current()[:, idx["D"]].sum()),
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

    # -- rigorous-analysis methods -------------------------------------------------------

    def _simulate(
        self, days: float, assemble_overrides: Optional[dict] = None, interventions: Optional[list] = None
    ) -> "Simulation":
        """Build a fresh clone from the stored assemble args (with overrides), apply the given
        interventions, advance it, and return it. Never mutates this session."""
        kwargs = dict(self._assemble_kwargs)
        if assemble_overrides:
            kwargs.update(assemble_overrides)
        clone = Simulation()
        clone.assemble(**kwargs)
        for iv in interventions or []:
            clone.apply_intervention(iv.kind, iv.magnitude, iv.start_day, iv.end_day)
        if days > 0:
            clone.advance(days)
        return clone

    def calibrate_r0(self, observed_days: list, observed_cumulative: list, low: float = 0.3, high: float = 8.0) -> dict:
        """Fit ``r0`` to observed cumulative-incidence data, holding other parameters fixed.

        Returns ``{"r0": fitted, "sse": residual_sum_of_squares}``. Grounds the transmission
        rate in data instead of assuming it. Non-mutating.
        """
        observed_days = list(observed_days)
        observed = np.array(observed_cumulative, dtype=float)
        horizon = max(observed_days)

        def sse(r0: float) -> float:
            times, cumulative = self._simulate(horizon, {"r0": float(r0)}, interventions=[])._series_at(observed_days)
            return float(np.sum((cumulative - observed) ** 2))

        result = minimize_scalar(sse, bounds=(low, high), method="bounded")
        return {"r0": float(result.x), "sse": float(result.fun)}

    def _series_at(self, days: list) -> tuple[list, np.ndarray]:
        """Interpolate this (already-advanced) session's cumulative infections at ``days``."""
        times = np.array(self._times)
        cumulative = self._series("C")
        return days, np.interp(days, times, cumulative)

    def optimize_intervention(
        self,
        kind: str,
        metric: str,
        threshold: float,
        days: float,
        low: float = 0.0,
        high: float = 1.0,
        max_iter: int = 26,
    ) -> dict:
        """Find the smallest magnitude of a single ongoing intervention (from day 0) that keeps
        ``metric`` at or below ``threshold`` over ``days``.

        Assumes the metric decreases monotonically with magnitude (true for the supported
        interventions and the burden/prevalence metrics). Returns
        ``{"magnitude", "value", "met"}``. Non-mutating; ignores currently-applied interventions.
        """
        if kind not in INTERVENTION_KINDS:
            raise ValueError(f"unknown intervention kind '{kind}'")

        def evaluate(magnitude: float) -> float:
            trial = Intervention(kind, magnitude, 0.0, None)
            return self._simulate(days, interventions=[trial]).metrics()[metric]

        if evaluate(low) <= threshold:
            return {"magnitude": low, "value": evaluate(low), "met": True}
        if evaluate(high) > threshold:
            return {"magnitude": high, "value": evaluate(high), "met": False}

        lo, hi = low, high
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if evaluate(mid) <= threshold:
                hi = mid
            else:
                lo = mid
        return {"magnitude": hi, "value": evaluate(hi), "met": True}

    def _ensemble_samples(
        self, days: float, n_samples: int, seed: int, spread: float, vary: Optional[list]
    ) -> tuple[list, np.ndarray, dict]:
        """Latin-hypercube sample the uncertain parameters around their assembled values, run
        the model (keeping the current interventions) for each, and collect output metrics."""
        if vary is None:
            transmission = "biting_rate" if self.config.include_vector else "r0"
            vary = [transmission, *_UNCERTAIN_PARAMETERS]

        base = {name: float(self._assemble_kwargs[name]) for name in vary}
        sampler = LatinHypercube(d=len(vary), seed=seed)
        unit = sampler.random(n_samples)  # (n_samples, d) in [0, 1)

        sample_matrix = np.zeros((n_samples, len(vary)))
        outputs = {metric: np.zeros(n_samples) for metric in _ENSEMBLE_METRICS}

        for row in range(n_samples):
            overrides = {}
            for col, name in enumerate(vary):
                value = base[name] * (1.0 - spread + 2.0 * spread * unit[row, col])
                overrides[name] = value
                sample_matrix[row, col] = value
            metrics = self._simulate(days, overrides, interventions=list(self.interventions)).metrics()
            for metric in _ENSEMBLE_METRICS:
                outputs[metric][row] = metrics[metric]

        return vary, sample_matrix, outputs

    def run_ensemble(
        self,
        days: float,
        n_samples: int = 100,
        seed: int = 0,
        spread: float = 0.2,
        vary: Optional[list] = None,
    ) -> dict:
        """Quantify outcome uncertainty by running an ensemble over plausible parameter values.

        Returns per-metric ``{median, mean, p05, p95}`` summaries. Reproducible for a given
        ``seed``. Keeps the currently-applied interventions fixed across the ensemble.
        """
        vary, _, outputs = self._ensemble_samples(days, n_samples, seed, spread, vary)
        summary = {}
        for metric, values in outputs.items():
            summary[metric] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        return {"n_samples": n_samples, "horizon_days": days, "parameters": vary, "summary": summary}

    def sensitivity_analysis(
        self,
        days: float,
        target: str = "peak_hospitalized",
        n_samples: int = 200,
        seed: int = 0,
        spread: float = 0.2,
        vary: Optional[list] = None,
    ) -> list:
        """Rank which parameters drive ``target`` via Spearman rank correlation over the
        ensemble samples. Returns a list of ``{"parameter", "correlation"}`` sorted by
        absolute correlation (descending)."""
        if target not in _ENSEMBLE_METRICS:
            raise ValueError(f"unknown target '{target}'; expected one of {_ENSEMBLE_METRICS}")
        vary, sample_matrix, outputs = self._ensemble_samples(days, n_samples, seed, spread, vary)
        target_values = outputs[target]

        drivers = []
        for col, name in enumerate(vary):
            inputs = sample_matrix[:, col]
            if np.allclose(inputs, inputs[0]):
                correlation = 0.0  # parameter did not vary (base value was 0)
            else:
                correlation = float(spearmanr(inputs, target_values).statistic)
                if np.isnan(correlation):
                    correlation = 0.0
            drivers.append({"parameter": name, "correlation": correlation})

        drivers.sort(key=lambda d: abs(d["correlation"]), reverse=True)
        return drivers

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
