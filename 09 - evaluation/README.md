# Evaluation

Benchmark models and agent configurations on the project's tasks, scoring local Ollama models, Anthropic models, and an MCP-tool-augmented agent against each other.

Part of the [genscai](../README.md) use-case series. Shared library code lives in the [`genscai`](../genscai) package; datasets in [`data/`](../data) and generated artifacts in `output/` are shared at the repo root.

## Notebooks

| # | Notebook |
| --- | --- |
| 01 | [01 - Evaluation - AIMU Benchmark](notebooks/01%20-%20Evaluation%20-%20AIMU%20Benchmark.ipynb) |

## Notes

Depends on `output/medrxiv.db` (built in `04 - semantic search`) and needs `ANTHROPIC_API_KEY` for the Anthropic client and the LLM-judge scorer.
