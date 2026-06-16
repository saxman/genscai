# Disease Simulation

An adaptive intervention-planning agent that drives the modular compartmental disease model in `genscai/simulation.py` — assembling a model, applying candidate interventions, and recommending a strategy.

Part of the [genscai](../README.md) use-case series. Shared library code lives in the [`genscai`](../genscai) package; datasets in [`data/`](../data) and generated artifacts in `output/` are shared at the repo root.

## Scripts

- [`01_intervention_agent.py`](scripts/01_intervention_agent.py)

## Notes

Set `GENSCAI_AGENT_MODEL` to choose the agent's model (defaults to an Anthropic model; can point at a local Ollama model).
