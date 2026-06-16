# Agents

Build tool-using agents: wire tools over the Model Context Protocol (MCP), use the smolagents framework, and compose multi-step AIMU workflows. For a focused comparison of agent frameworks on one task, see `07 - literature research/`.

Part of the [genscai](../README.md) use-case series. Shared library code lives in the [`genscai`](../genscai) package; datasets in [`data/`](../data) and generated artifacts in `output/` are shared at the repo root.

## Notebooks

| # | Notebook |
| --- | --- |
| 01 | [01 - Agents - MCP Tools](notebooks/01%20-%20Agents%20-%20MCP%20Tools.ipynb) |
| 02 | [02 - Agents - smolagents](notebooks/02%20-%20Agents%20-%20smolagents.ipynb) |
| 03 | [03 - Agents - Paperclip](notebooks/03%20-%20Agents%20-%20Paperclip.ipynb) |
| 04 | [04 - Agents - AIMU Workflows](notebooks/04%20-%20Agents%20-%20AIMU%20Workflows.ipynb) |

## Scripts

- [`01_agent.py`](scripts/01_agent.py)

## Notes

The MCP and workflow notebooks query `output/medrxiv.db`; build it first via the `04 - semantic search` scripts.
