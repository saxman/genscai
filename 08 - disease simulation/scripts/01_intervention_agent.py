"""Adaptive intervention-planning agent over a modular compartmental model.

An AIMU agent is given an outbreak scenario and tools that let it operate a disease-modeling
session (`genscai.simulation.Simulation`). The intended workflow is:

    diagnose the scenario -> assemble a fit-for-purpose model -> CALIBRATE transmission to
    observed case data -> run a no-intervention baseline -> size interventions with a real
    optimizer -> quantify outcome UNCERTAINTY and parameter SENSITIVITY -> recommend a
    strategy, reported as ranges with explicit assumptions and limitations.

The engine always integrates the full coupled SEIR(+hospitalization) system, so the agent's
structural choice is *which mechanism modules to include* (mixing structure, optional
host-vector dynamics), not which equations to advance.

IMPORTANT - intended use and limitations
-----------------------------------------
This is a TEACHING DEMONSTRATION of how to wire an LLM agent to a quantitative model so it
calibrates to data, sizes interventions with an optimizer, and reports uncertainty - i.e. how
to use AI *responsibly* around an epidemiological model. It is NOT a validated public-health
tool and must not be used for real decisions:

- The compartmental engine is a simplified teaching model, not peer-reviewed (a real study
  would use a framework such as Starsim, EMOD, or LASER).
- All epidemiological parameters here are illustrative; only R0 is fit to (synthetic) data.
- For a deterministic model the optimum is directly computable, so the LLM adds interface,
  orchestration, and uncertainty-communication value - not numerical accuracy.
- Output is decision-support for expert review, never a directive. Accountability stays human.

The tools below close over a single module-level `sim`: `@aimu.tool` requires plain functions
with primitive parameters, so a shared session object is the simplest place for the stateless
tools to act. This is a single-run demo convenience, not a pattern for concurrent use.

Run with::

    python "08 - disease simulation/scripts/01_intervention_agent.py"

Requires `ANTHROPIC_API_KEY` for the default model. Override the model with the
`GENSCAI_AGENT_MODEL` env var (e.g. `ollama:qwen3.5:9b` for a local tools-capable model).
"""

import json
import os
from datetime import datetime

import aimu
from aimu.models import StreamingContentType
from dotenv import load_dotenv

from genscai import paths
from genscai.simulation import INTERVENTION_KINDS, Simulation

load_dotenv()

MODEL = os.environ.get("GENSCAI_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

# Reproducibility: the ensemble and sensitivity analyses are seeded so a given run is
# repeatable. The deterministic engine is itself reproducible; only the sampling needs a seed.
ANALYSIS_SEED = int(os.environ.get("GENSCAI_ANALYSIS_SEED", "20240615"))
ENSEMBLE_SAMPLES = 60
SENSITIVITY_SAMPLES = 120

HORIZON_DAYS = 365
HOSPITAL_CAPACITY = 2_500  # regional acute-care beds

# Observed early cumulative case counts (illustrative). The agent calibrates R0 to these
# rather than being told it.
OBSERVED_DAYS = [7, 14, 21, 28]
OBSERVED_CASES = [195, 496, 1041, 2029]

SCENARIO = f"""\
A novel pathogen is spreading in a region of three connected cities (roughly equal
populations, ~1,000,000 people total). Surveillance to date:

- Cases first appeared in one city; people commute daily between the three cities.
- Estimated incubation period ~5 days; infectious period ~8 days.
- About 4% of infections require hospitalization; without crisis-level overload roughly 15%
  of hospitalized patients die. The pathogen spreads person-to-person (not vector-borne).
- The region has {HOSPITAL_CAPACITY:,} acute-care beds total.
- Observed cumulative confirmed cases so far: {", ".join(f"day {d}: {c:,}" for d, c in zip(OBSERVED_DAYS, OBSERVED_CASES))}.

Do NOT assume a value for R0 - calibrate it to the observed case counts.

Objective: recommend an intervention strategy that keeps peak hospital demand at or below the
{HOSPITAL_CAPACITY:,}-bed capacity (so the health system is not overwhelmed and crisis
mortality is avoided) and minimizes deaths over a {HORIZON_DAYS}-day horizon, while preferring
less disruptive interventions where they suffice.
"""

SYSTEM_PROMPT = f"""\
You are an infectious-disease modeler operating a simulation through tools. You are producing
DECISION-SUPPORT for human experts to review, not a directive. Work rigorously:

1. `describe_available_dynamics`, then choose the mixing structure (and whether vectors apply).
2. `assemble_model` with the scenario's structural and burden parameters and a provisional R0.
3. `calibrate_transmission` on the observed case data to fit R0 BEFORE drawing conclusions.
   Never assume R0; ground it in data.
4. Establish a no-intervention baseline: `advance_simulation` to the horizon, `get_metrics`.
   Note peak hospitalizations vs. capacity and projected deaths.
5. Size interventions with `optimize_intervention` (a real search) instead of guessing
   magnitudes. Refine and combine interventions, using `reset_simulation` between trials.
6. Before finalizing, ALWAYS run `run_uncertainty_analysis` on your chosen strategy and
   `run_sensitivity_analysis` to see which parameters drive the outcome.

Intervention semantics (the `magnitude` argument is interpreted per kind):
- vaccination: fraction of remaining susceptibles immunized per day (capped at 80% coverage).
- contact_reduction: fractional cut in transmission (0.5 halves the transmission rate).
- isolation: fractional cut in how much infectious people contribute to new infections.
- treatment: fractional increase in the recovery rate (shortens the infectious period).
Available intervention kinds: {", ".join(INTERVENTION_KINDS)}.

Act, do not just plan. In any turn where the work is not finished, END THE TURN BY CALLING A
TOOL. Never end a turn with only a description of what you intend to do next. Reserve plain
prose with no tool call for your FINAL recommendation.

Your final recommendation MUST:
- LEAD with limitations and the key assumptions (illustrative parameters, simplified model,
  only R0 calibrated, deterministic-model caveats).
- Report outcomes as RANGES from the uncertainty analysis (median with 5th-95th percentile),
  not single numbers. Do not imply false precision.
- Distinguish what "the model projects" from what "will happen".
- Recommend the least-disruptive strategy that keeps projected peak hospital demand within
  capacity across the uncertainty range, and state explicitly that it is for expert review.
"""

sim = Simulation()


def _format_metrics(metrics: dict) -> str:
    population = sim.config.population if sim.config else 1
    capacity = metrics["hospital_capacity"]
    lines = [
        f"day: {metrics['day']:.0f}",
        f"peak infectious: {metrics['peak_infectious']:,.0f} "
        f"({metrics['peak_infectious'] / population:.1%} of population) on day {metrics['peak_day']:.0f}",
        f"peak hospitalized: {metrics['peak_hospitalized']:,.0f} on day {metrics['peak_hospital_day']:.0f} "
        f"(capacity {capacity:,.0f}; days over capacity: {metrics['days_over_capacity']})",
        f"cumulative infections: {metrics['cumulative_infections']:,.0f} "
        f"(attack rate {metrics['attack_rate']:.1%})",
        f"cumulative deaths: {metrics['cumulative_deaths']:,.0f}",
    ]
    if metrics.get("r_effective") is not None:
        lines.append(f"current R_effective: {metrics['r_effective']:.2f}")
    if "vector_infectious" in metrics:
        lines.append(f"infectious vectors: {metrics['vector_infectious']:,.0f}")
    return "\n".join(lines)


@aimu.tool
def describe_available_dynamics() -> str:
    """List the mechanism modules the model can be assembled from and when each fits."""
    return (
        "Mixing structures (pick one):\n"
        "- mass_action: one well-mixed population. Use for a single homogeneous community.\n"
        "- metapopulation: several patches (e.g. cities) coupled by migration between them. "
        "Use when the outbreak spans connected sub-populations with travel. Set num_patches "
        "and migration_rate (per-day mixing toward the cross-patch average).\n"
        "- network: heterogeneous contact density. Use when contact intensity differs from a "
        "typical community. Set mean_contacts (contacts/day; 10 is the neutral baseline, "
        "higher spreads faster).\n\n"
        "Optional host-vector module (include_vector=True): transmission flows host->vector->host "
        "via biting (set biting_rate). Use for vector-borne diseases (e.g. mosquito-borne). When "
        "enabled, r0 no longer drives transmission; biting_rate does.\n\n"
        "All models track hospitalizations and deaths; deaths rise when hospital occupancy "
        "exceeds capacity (crisis-care effect)."
    )


@aimu.tool
def assemble_model(
    mixing: str,
    population: int,
    initial_infected: int,
    r0: float,
    incubation_days: float,
    infectious_days: float,
    num_patches: int = 1,
    migration_rate: float = 0.0,
    mean_contacts: float = 10.0,
    include_vector: bool = False,
    biting_rate: float = 0.0,
    hospital_capacity: float = 0.0,
    hospitalized_fraction: float = 0.05,
    hospital_fatality_fraction: float = 0.15,
) -> str:
    """Build the disease model from the chosen mechanism modules and epidemiological parameters.

    Args:
        mixing: "mass_action", "metapopulation", or "network".
        population: total host population across all patches.
        initial_infected: number initially infectious, seeded in the first patch.
        r0: provisional basic reproduction number (calibrate it next; ignored if include_vector).
        incubation_days: average days from infection to becoming infectious.
        infectious_days: average days a host remains infectious.
        num_patches: number of patches (metapopulation mixing).
        migration_rate: per-day mixing rate between patches (metapopulation mixing).
        mean_contacts: average contacts/day (network mixing; 10 is the neutral baseline).
        include_vector: include host-vector transmission dynamics.
        biting_rate: vector biting/transmission intensity per day (when include_vector is True).
        hospital_capacity: total acute-care beds; 0 lets the model pick ~2.5 beds per 1000.
        hospitalized_fraction: fraction of infections requiring hospitalization.
        hospital_fatality_fraction: fraction of hospitalized patients who die (below capacity).
    """
    try:
        sim.assemble(
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
            hospital_capacity=hospital_capacity,
            hospitalized_fraction=hospitalized_fraction,
            hospital_fatality_fraction=hospital_fatality_fraction,
        )
    except (ValueError, ZeroDivisionError) as error:
        return f"Error assembling model: {error}"

    state = sim.state()
    return (
        f"Model assembled: {mixing} mixing, {population:,} people"
        + (f" across {num_patches} patches" if num_patches > 1 else "")
        + (", host-vector dynamics enabled" if include_vector else "")
        + f", hospital capacity {sim.config.hospital_capacity:,.0f}."
        + f" Initial state at day 0 - susceptible: {state['susceptible']:,.0f}, "
        f"infectious: {state['infectious']:,.0f}."
    )


@aimu.tool
def calibrate_transmission(observed_days: list, observed_cumulative_cases: list) -> str:
    """Fit R0 to observed cumulative case counts on the current model, then adopt the fit.

    Holds all other parameters fixed and updates the model to the fitted R0 (resetting to day 0).
    Call this after assemble_model and before drawing any conclusions.

    Args:
        observed_days: days at which cumulative cases were observed (e.g. [7, 14, 21, 28]).
        observed_cumulative_cases: cumulative confirmed cases at those days.
    """
    if sim.config is None:
        return "Error: assemble a model first (with a provisional r0); calibration fits R0 on it."
    try:
        days = [float(d) for d in observed_days]
        cases = [float(c) for c in observed_cumulative_cases]
        result = sim.calibrate_r0(days, cases)
    except (ValueError, TypeError) as error:
        return f"Error during calibration: {error}"
    sim.reconfigure(r0=result["r0"])
    return (
        f"Calibrated R0 = {result['r0']:.2f} (fit residual SSE {result['sse']:,.0f}). "
        "Model updated to the fitted R0 and reset to day 0."
    )


@aimu.tool
def apply_intervention(kind: str, magnitude: float, start_day: float, end_day: float = -1.0) -> str:
    """Apply a time-windowed intervention to the current model.

    Args:
        kind: one of vaccination, contact_reduction, isolation, treatment.
        magnitude: strength of the intervention (interpreted per kind; see the system prompt).
        start_day: day the intervention begins.
        end_day: day the intervention ends; use -1 for an open-ended (permanent) intervention.
    """
    if sim.config is None:
        return "Error: assemble a model before applying interventions."
    try:
        sim.apply_intervention(kind, magnitude, start_day, None if end_day < 0 else end_day)
    except ValueError as error:
        return f"Error: {error}"
    window = "ongoing" if end_day < 0 else f"until day {end_day:.0f}"
    return f"Applied {kind} (magnitude {magnitude}) from day {start_day:.0f}, {window}."


@aimu.tool
def list_interventions() -> str:
    """List the interventions currently applied to the model."""
    if not sim.interventions:
        return "No interventions currently applied."
    lines = []
    for i, iv in enumerate(sim.interventions):
        window = "ongoing" if iv.end_day is None else f"until day {iv.end_day:.0f}"
        lines.append(f"[{i}] {iv.kind} (magnitude {iv.magnitude}) from day {iv.start_day:.0f}, {window}")
    return "\n".join(lines)


@aimu.tool
def remove_intervention(index: int) -> str:
    """Remove the intervention at the given index (see list_interventions)."""
    try:
        removed = sim.remove_intervention(index)
    except IndexError:
        return f"Error: no intervention at index {index}."
    return f"Removed {removed.kind} at index {index}."


@aimu.tool
def advance_simulation(days: float) -> str:
    """Integrate the simulation forward by the given number of days and report the result."""
    if sim.config is None:
        return "Error: assemble a model before advancing the simulation."
    try:
        sim.advance(days)
    except RuntimeError as error:
        return f"Error advancing simulation: {error}"
    return f"Advanced {days:.0f} days (now at day {sim.day:.0f}).\n{_format_metrics(sim.metrics())}"


@aimu.tool
def get_metrics() -> str:
    """Report current outcome metrics: prevalence, hospital demand vs capacity, deaths, R_effective."""
    if sim.config is None:
        return "Error: assemble a model first."
    return _format_metrics(sim.metrics())


@aimu.tool
def optimize_intervention(kind: str, metric: str, threshold: float, days: float, low: float = 0.0, high: float = 1.0) -> str:
    """Find the smallest magnitude of a single ongoing intervention that keeps a metric within a target.

    Searches for the minimal magnitude of `kind` (applied alone, from day 0) such that `metric`
    stays at or below `threshold` over `days`. Use this to size interventions instead of guessing.

    Args:
        kind: one of vaccination, contact_reduction, isolation, treatment.
        metric: outcome to bound; one of peak_infectious, peak_hospitalized, cumulative_deaths,
            days_over_capacity, attack_rate.
        threshold: the maximum acceptable value of the metric.
        days: horizon over which to evaluate.
        low: lower bound of the magnitude search (e.g. 0).
        high: upper bound of the magnitude search (e.g. 0.9; for vaccination try ~0.05).
    """
    if sim.config is None:
        return "Error: assemble a model first."
    try:
        result = sim.optimize_intervention(kind, metric, threshold, days, low, high)
    except ValueError as error:
        return f"Error: {error}"
    verb = "meets" if result["met"] else "CANNOT meet (within the search bounds)"
    return (
        f"Smallest {kind} magnitude that {verb} {metric} <= {threshold:,.0f} over {days:.0f} days: "
        f"{result['magnitude']:.4f}, achieving {metric} = {result['value']:,.0f}. "
        "(Evaluated as the only intervention, applied from day 0.)"
    )


@aimu.tool
def run_uncertainty_analysis(days: float) -> str:
    """Quantify outcome uncertainty for the current model and interventions over a parameter ensemble.

    Reports median and 5th-95th percentile ranges across plausible parameter values. Run this on
    your chosen strategy before finalizing, and report ranges rather than single numbers.
    """
    if sim.config is None:
        return "Error: assemble a model first."
    result = sim.run_ensemble(days, n_samples=ENSEMBLE_SAMPLES, seed=ANALYSIS_SEED)
    summary = result["summary"]

    def interval(label: str, key: str) -> str:
        stat = summary[key]
        return f"{label}: {stat['median']:,.0f} [{stat['p05']:,.0f} - {stat['p95']:,.0f}]"

    attack = summary["attack_rate"]
    lines = [
        f"Ensemble of {result['n_samples']} runs (parameters {', '.join(result['parameters'])} "
        f"varied +/-20%), {days:.0f}-day horizon. Median [5th - 95th percentile]:",
        interval("peak infectious", "peak_infectious"),
        interval("peak hospitalized", "peak_hospitalized"),
        f"(hospital capacity is {sim.config.hospital_capacity:,.0f})",
        interval("cumulative deaths", "cumulative_deaths"),
        f"attack rate: {attack['median']:.1%} [{attack['p05']:.1%} - {attack['p95']:.1%}]",
        interval("days over capacity", "days_over_capacity"),
    ]
    return "\n".join(lines)


@aimu.tool
def run_sensitivity_analysis(days: float, target: str = "peak_hospitalized") -> str:
    """Rank which uncertain parameters most drive an outcome (Spearman rank correlation).

    Args:
        days: horizon over which to evaluate.
        target: outcome to analyze; one of peak_infectious, peak_hospitalized,
            cumulative_deaths, attack_rate, days_over_capacity.
    """
    if sim.config is None:
        return "Error: assemble a model first."
    try:
        drivers = sim.sensitivity_analysis(days, target=target, n_samples=SENSITIVITY_SAMPLES, seed=ANALYSIS_SEED)
    except ValueError as error:
        return f"Error: {error}"
    lines = [f"Sensitivity of {target} ({SENSITIVITY_SAMPLES} samples, Spearman rank correlation):"]
    for driver in drivers:
        lines.append(f"  {driver['parameter']}: {driver['correlation']:+.2f}")
    return "\n".join(lines)


@aimu.tool
def checkpoint(label: str) -> str:
    """Save the current simulation state under a label so it can be restored later."""
    if sim.config is None:
        return "Error: assemble a model first."
    sim.checkpoint(label)
    return f"Checkpoint '{label}' saved at day {sim.day:.0f}."


@aimu.tool
def rewind(label: str) -> str:
    """Restore a previously saved checkpoint."""
    try:
        sim.rewind(label)
    except KeyError:
        return f"Error: no checkpoint named '{label}'."
    return f"Rewound to checkpoint '{label}' (day {sim.day:.0f})."


@aimu.tool
def reset_simulation() -> str:
    """Return to day 0 keeping the assembled model, and clear all interventions to try another."""
    if sim.config is None:
        return "Error: assemble a model first."
    sim.reset()
    return "Simulation reset to day 0; all interventions cleared. The model structure is unchanged."


TOOLS = [
    describe_available_dynamics,
    assemble_model,
    calibrate_transmission,
    apply_intervention,
    list_interventions,
    remove_intervention,
    advance_simulation,
    get_metrics,
    optimize_intervention,
    run_uncertainty_analysis,
    run_sensitivity_analysis,
    checkpoint,
    rewind,
    reset_simulation,
]

LIMITATIONS_BANNER = (
    "=" * 80 + "\n"
    "TEACHING DEMONSTRATION - not a validated public-health tool.\n"
    "Parameters are illustrative; only R0 is calibrated (to synthetic data). Output is\n"
    "decision-support for expert review, never a directive.\n" + "=" * 80
)


def _write_provenance(provenance: list, final_text: str) -> str:
    paths.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "timestamp": timestamp,
        "model": MODEL,
        "analysis_seed": ANALYSIS_SEED,
        "scenario": SCENARIO,
        "tool_calls": provenance,
        "final_recommendation": final_text,
    }
    out_path = paths.output / f"intervention_agent_run_{timestamp}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return str(out_path)


def run_agent() -> None:
    print(LIMITATIONS_BANNER)

    agent = aimu.agent(
        MODEL,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        max_iterations=32,
        final_answer_prompt=(
            "Give your final recommendation. Lead with limitations and assumptions, report "
            "outcomes as ranges from the uncertainty analysis (not single numbers), distinguish "
            "what the model projects from what will happen, and state that this is decision-support "
            "for expert review."
        ),
    )

    provenance: list = []
    final_text: list = []
    last_phase = None
    for chunk in agent.run(SCENARIO, stream=True):
        if chunk.is_tool_call():
            call = chunk.content
            response = str(call["response"]).strip()
            arguments = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
            print(f"\n\n[tool] {call['name']}({arguments})")
            print(f"[result] {response}\n")
            provenance.append({"tool": call["name"], "arguments": call["arguments"], "result": response})
            last_phase = None
            continue
        if chunk.is_text():
            if chunk.phase != last_phase:
                header = "thinking" if chunk.phase == StreamingContentType.THINKING else "agent"
                print(f"\n\n--- {header} ---")
                last_phase = chunk.phase
            if chunk.phase == StreamingContentType.GENERATING:
                final_text.append(chunk.content)
            print(chunk.content, end="", flush=True)

    provenance_path = _write_provenance(provenance, "".join(final_text))
    print(f"\n\nProvenance written to {provenance_path}")


if __name__ == "__main__":
    run_agent()
