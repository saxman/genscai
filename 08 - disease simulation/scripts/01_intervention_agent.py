"""Adaptive intervention-planning agent over a modular compartmental model.

An AIMU agent is given an outbreak scenario and a set of tools that let it operate a
disease-modeling session (`genscai.simulation.Simulation`). The intended loop is:

    diagnose the scenario -> assemble a fit-for-purpose model -> run a no-intervention
    baseline -> apply a candidate intervention -> evaluate it -> rewind and try another
    if it underperforms -> recommend a strategy.

The simulation engine always integrates the full coupled SEIR system, so the agent's
structural choice is *which mechanism modules to include* (mixing structure, optional
host-vector dynamics), not which equations to advance.

The tools below close over a single module-level `sim`. This is a deliberate simplification
for a single-run demo: `@aimu.tool` requires plain functions with primitive parameters, so a
shared session object is the simplest way to give the stateless tools somewhere to act.

Run with::

    python scripts/09_intervention_agent.py

Requires `ANTHROPIC_API_KEY` for the default model. Override the model with the
`GENSCAI_AGENT_MODEL` env var (e.g. `ollama:qwen3.5:9b` for a local tools-capable model).
"""

import os

import aimu
from aimu.models import StreamingContentType
from dotenv import load_dotenv

from genscai.simulation import INTERVENTION_KINDS, Simulation

load_dotenv()

MODEL = os.environ.get("GENSCAI_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

PEAK_PREVALENCE_TARGET = 0.05  # keep peak infectious below 5% of the population
HORIZON_DAYS = 365

SCENARIO = f"""\
A novel pathogen is spreading in a region of three connected cities (roughly equal
populations, ~1,000,000 people total). Early surveillance suggests:

- A basic reproduction number (R0) around 2.6.
- An incubation period of about 5 days and an infectious period of about 8 days.
- Cases first appeared in one city and are now appearing in the others; people commute
  between the cities daily.
- The pathogen spreads person-to-person (it is not vector-borne).

Your objective: recommend an intervention strategy that keeps peak infectious prevalence
below {PEAK_PREVALENCE_TARGET:.0%} of the population over a {HORIZON_DAYS}-day horizon, while
preferring less disruptive interventions where possible.

Work the problem with your tools: choose the model structure that fits this scenario,
establish a no-intervention baseline first, then test interventions, rewinding to try
alternatives when one does not meet the objective.
"""

SYSTEM_PROMPT = f"""\
You are an infectious-disease modeler operating a simulation through tools. Follow this loop:

1. Call `describe_available_dynamics` to review the mechanism modules, then choose the
   mixing structure (and whether to include vector dynamics) that fits the scenario.
2. Call `assemble_model` to build the model.
3. ALWAYS establish a no-intervention baseline first: `advance_simulation` to the full
   horizon and read `get_metrics`. This is the comparison point for every intervention.
4. `reset_simulation`, then apply one or more interventions and advance again.
5. Judge effectiveness against the objective. An intervention strategy is effective if it
   keeps peak infectious prevalence below the stated target and improves on the baseline.
   If it does not, `reset_simulation` and modify the magnitude or timing, or try a
   different intervention. Do not give up after a single attempt.

Intervention semantics (the `magnitude` argument is interpreted per kind):
- vaccination: fraction of remaining susceptibles immunized per day (capped at 80% coverage).
- contact_reduction: fractional cut in transmission (0.5 halves the transmission rate).
- isolation: fractional cut in how much infectious people contribute to new infections.
- treatment: fractional increase in the recovery rate (shortens the infectious period).

Available intervention kinds: {", ".join(INTERVENTION_KINDS)}.

Act, do not just plan. In any turn where the objective is not yet met, END THE TURN BY
CALLING A TOOL (assemble_model, apply_intervention, advance_simulation, reset_simulation,
get_metrics, ...). Never end a turn with only a description of what you intend to do next;
issue the tool call instead. Reserve plain prose with no tool call for your FINAL
recommendation, once the objective is met or you have exhausted reasonable options.

Be systematic and concise. Your final recommendation should state the chosen model structure,
the interventions you tried with their key metrics, and your recommended strategy.
"""

sim = Simulation()


def _format_metrics(metrics: dict) -> str:
    population = sim.config.population if sim.config else 1
    lines = [
        f"day: {metrics['day']:.0f}",
        f"infectious now: {metrics['infectious']:,.0f}",
        f"peak infectious: {metrics['peak_infectious']:,.0f} "
        f"({metrics['peak_infectious'] / population:.1%} of population) on day {metrics['peak_day']:.0f}",
        f"cumulative infections: {metrics['cumulative_infections']:,.0f} "
        f"(attack rate {metrics['attack_rate']:.1%})",
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
        "enabled, r0 no longer drives transmission; biting_rate does."
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
) -> str:
    """Build the disease model from the chosen mechanism modules and epidemiological parameters.

    Args:
        mixing: "mass_action", "metapopulation", or "network".
        population: total host population across all patches.
        initial_infected: number initially infectious, seeded in the first patch.
        r0: basic reproduction number (ignored when include_vector is True).
        incubation_days: average days from infection to becoming infectious.
        infectious_days: average days a host remains infectious.
        num_patches: number of patches (metapopulation mixing).
        migration_rate: per-day mixing rate between patches (metapopulation mixing).
        mean_contacts: average contacts/day (network mixing; 10 is the neutral baseline).
        include_vector: include host-vector transmission dynamics.
        biting_rate: vector biting/transmission intensity per day (when include_vector is True).
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
        )
    except (ValueError, ZeroDivisionError) as error:
        return f"Error assembling model: {error}"

    state = sim.state()
    return (
        f"Model assembled: {mixing} mixing, {population:,} people"
        + (f" across {num_patches} patches" if num_patches > 1 else "")
        + (", host-vector dynamics enabled" if include_vector else "")
        + f". Initial state at day 0 - susceptible: {state['susceptible']:,.0f}, "
        f"infectious: {state['infectious']:,.0f}."
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
    """Report current outcome metrics: peak prevalence, attack rate, cumulative cases, R_effective."""
    if sim.config is None:
        return "Error: assemble a model first."
    return _format_metrics(sim.metrics())


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
    apply_intervention,
    list_interventions,
    remove_intervention,
    advance_simulation,
    get_metrics,
    checkpoint,
    rewind,
    reset_simulation,
]


def run_agent() -> None:
    agent = aimu.agent(
        MODEL,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        max_iterations=25,
        final_answer_prompt=(
            "Summarize the model structure you chose, the interventions you tried with their key "
            "metrics, and your recommended intervention strategy with the metrics that support it."
        ),
    )

    last_phase = None
    for chunk in agent.run(SCENARIO, stream=True):
        if chunk.is_tool_call():
            call = chunk.content
            arguments = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
            print(f"\n\n[tool] {call['name']}({arguments})")
            print(f"[result] {str(call['response']).strip()}\n")
            last_phase = None
            continue
        if chunk.is_text():
            if chunk.phase != last_phase:
                header = "thinking" if chunk.phase == StreamingContentType.THINKING else "agent"
                print(f"\n\n--- {header} ---")
                last_phase = chunk.phase
            print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    run_agent()
