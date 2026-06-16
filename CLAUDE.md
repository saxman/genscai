# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (uv recommended)
uv venv && source .venv/bin/activate && uv sync --all-extras

# Lint
ruff check .

# Test
pytest

# Run chatbot
streamlit run streamlit/genscai_chatbot.py

# Run notebooks
jupyter lab
```

`--all-extras` covers dev, notebooks, and CUDA groups. On Windows use `.venv\Scripts\activate`.

## Use-case folders

The project is organized into top-level **use-case folders**, each containing its own `notebooks/`
and/or `scripts/` subdir. The shared package (`genscai/`), datasets (`data/`), generated artifacts
(`output/`), and tests (`tests/`) live at the root and are imported/referenced from any folder via
`genscai.paths` (anchored to the package, so notebooks/scripts run from any depth). Folders:

| Folder | Contents |
|---|---|
| `00 - getting started/` | notebook `01` (Environment Setup) |
| `01 - data collection/` | notebooks `01`–`03`; script `01_retrieve_information.py` |
| `02 - text analysis/` | notebooks `01`–`10` (information extraction, embeddings, classification); scripts `01`–`05`, `papers_json_to_csv.py` |
| `03 - knowledge graphs/` | notebooks `01`, `02` (GraphRAG, LangChain) |
| `04 - semantic search/` | notebooks `01`, `02` (RAG pipeline, RAG evaluation); scripts `01_medrxiv_download.py`, `02_medrxiv_knowledge_base.py` |
| `05 - model optimization/` | notebooks `01`, `02` |
| `06 - agents/` | notebooks `01`–`04`; script `01_agent.py` |
| `07 - literature research/` | notebooks `01`–`07` (AIMU, genscai-corpus, then five frameworks) + `README.md` (see below) |
| `08 - disease simulation/` | script `01_intervention_agent.py` (drives `genscai/simulation.py`) |
| `09 - evaluation/` | notebook `01` (AIMU Benchmark) |

Each folder is numbered (use-case names have no hyphens), has a `README.md`, and both its notebooks
and scripts are renumbered from `01`. The medRxiv download/index logic lives in
`genscai/knowledge_base.py`; the `04 - semantic search/` scripts are thin CLIs over it.

`07 - literature research/` implements one identical use case — agentic literature research over
medRxiv/bioRxiv with a relevance-gated local document store and a critic feedback loop — six ways
across frameworks (`01` AIMU, `03` smolagents, `04` LangGraph, `05` PydanticAI, `06` CrewAI, `07`
LlamaIndex) plus `02` genscai, a "batteries-included" build on the project's own
`genscai.corpus.Corpus` helper (layered on AIMU). All run on local Ollama models; the notebooks are
meant to be compared, not sequenced (each script is a runnable counterpart). Shared search/fetch
tools live in `genscai/research.py`; install with `uv sync --all-extras` (the `literature_research`
extra). Default model `qwen3.6:27b`, overridable via `GENSCAI_AGENT_MODEL`. See
`07 - literature research/README.md`.

`02 - genscai` exercises AIMU capabilities added for this use case: `ToolContext` dependency
injection (tools reach shared state without globals), `EvaluatorOptimizer(verdict_schema=...)` (a
typed critic verdict), and `aimu.pretty_print`. `01 - AIMU` uses the same three directly, by hand.

The RAG notebooks (`04 - semantic search/notebooks/01` Pipeline and `02` Evaluation) depend on `output/medrxiv.db` built by
`04 - semantic search/scripts/02_medrxiv_knowledge_base.py` (or `genscai.knowledge_base.build_knowledge_base`). The RAG Evaluation notebook also requires `pip install ragas datasets`.

`06 - agents/notebooks/04` (AIMU Workflows) and `09 - evaluation/notebooks/01` (AIMU Benchmark) also depend on `output/medrxiv.db` because they wire the genscai MCP `search_research_articles` tool into agent and benchmark clients. The benchmark additionally needs `ANTHROPIC_API_KEY` set for the Anthropic client and the `LLMJudgeScorer` judge.

## Architecture

**GenScAI** is a research toolkit that pipelines literature into a searchable, agent-accessible knowledge base. The flow is:

```
MIDAS website / medRxiv API
  → 01 - data collection/scripts/01 & 04 - semantic search/scripts/01  (scrape / download raw JSON + TinyDB)
  → 02 - text analysis/scripts/02 & 04   (prompt optimization → paper classification → modeling_papers.json)
  → 04 - semantic search/scripts/02      (index into Chroma vector DB: output/medrxiv.db; logic in genscai/knowledge_base.py)
  → 03 - knowledge graphs/ & 04 - semantic search/notebooks/  (knowledge graphs, RAG, RAG evaluation)
  → genscai/tools.py                     (MCP server exposing search_research_articles)
  → streamlit chatbot                    (multi-LLM UI wired to MCP tools)
```

### `genscai/` package modules

| Module | Role |
|---|---|
| `paths.py` | Single source of truth for `root`, `data`, `output`, `tests`, `package` paths |
| `retrieval.py` | `MIDASRetriever` — scrapes midasnetwork.us, stores to TinyDB |
| `medrxiv.py` | medRxiv REST API client (`retrieve_articles`) |
| `knowledge_base.py` | Download medRxiv articles and build/index the Chroma knowledge base (used by `04 - semantic search/` scripts and `tools.py`) |
| `research.py` | Live literature search/fetch via Europe PMC + arXiv (used by `07 - literature research/`) |
| `corpus.py` | `Corpus` — relevance-gated research corpus exposing search/save/read as AIMU agent tools over a `SemanticMemoryStore` (used by `07 - literature research/` notebook/script `02`) |
| `classification.py` | LLM-based paper classifier; parses `[YES]`/`[NO]` responses; exports metrics |
| `training.py` | Prompt mutation templates for hill-climbing prompt optimization |
| `simulation.py` | Modular compartmental disease-modeling engine (used by `08 - disease simulation/`) |
| `data.py` | Loaders for the labeled training set and MIDAS Hugging Face dataset |
| `tools.py` | FastMCP server — exposes `search_research_articles` over the MCP protocol |
| `utils.py` | `ReadOnlyTinyDB`, date extraction from paths, CUDA device info |

### Scripts pipeline

Scripts are renumbered from `01` within each use-case folder's `scripts/` subdir (listed here in
overall execution order):

- `01 - data collection/scripts/01_retrieve_information.py` — Retrieve MIDAS abstracts; deduplicate; split 70/15/15 into train/test/validate CSVs
- `02 - text analysis/scripts/01_classification_test.py` — Spot-check classification quality against labeled data
- `02 - text analysis/scripts/02_classification_training.py` — Auto-tune classification prompts via hill-climbing mutation; saves best prompts to `data/prompt_catalog.db` (TinyDB)
- `02 - text analysis/scripts/03_classification_cross_model_validation.py` — Cross-model validation: test prompts tuned on one model against others
- `02 - text analysis/scripts/04_classification.py` — Classify all MIDAS papers; output `data/modeling_papers.json`
- `02 - text analysis/scripts/05_classification_all_models.py` — Same classification across multiple open models in parallel
- `02 - text analysis/scripts/papers_json_to_csv.py` — Convert classification results to modeling/non-modeling CSVs
- `04 - semantic search/scripts/01_medrxiv_download.py` — Download medRxiv papers for 2019–2024 to `output/medrxiv_{year}.json` (thin CLI over `genscai.knowledge_base`)
- `04 - semantic search/scripts/02_medrxiv_knowledge_base.py` — Index medRxiv abstracts into Chroma (`output/medrxiv.db`), 256-char chunks with 50-char overlap (thin CLI over `genscai.knowledge_base`)
- `06 - agents/scripts/01_agent.py` — Minimal smolagents example (web search agent)
- `08 - disease simulation/scripts/01_intervention_agent.py` — Adaptive intervention-planning agent over `genscai/simulation.py`

### LLM abstraction

All LLM calls go through **`aimu`** (a thin unified client over Ollama, HuggingFace Transformers, and Anthropic). The streamlit chatbot also supports **`litellm`** for broader model routing. **`smolagents`** handles agentic workflows; **FastMCP** exposes tools over the Model Context Protocol.

### Classification prompt engineering

`classification.py` contains three prompt templates:
- `CLASSIFICATION_TASK_PROMPT_TEMPLATE` — defines the classification task
- `CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE` — enforces `[YES]`/`[NO]` output format
- Generation kwargs default to `max_new_tokens=3, temperature=0.01` for deterministic classification

`training.py` adds mutation templates for positive/negative examples used during prompt hill-climbing in `02 - text analysis/scripts/02_classification_training.py`.

### Streamlit chatbot

`streamlit/genscai_chatbot.py` connects to four MCP servers at runtime:
- `genscai` (local `tools.py`) — research article search
- `laser_core`, `laser_generic` — LASER disease modeling framework docs
- `starsim_core` — Starsim disease modeling framework docs

It streams responses and renders tool calls and model thinking in `st.expander` blocks.

### Key data locations

| Path | Contents |
|---|---|
| `data/` | Labeled training CSVs, `prompt_catalog.db`, `modeling_papers*.json` |
| `output/` | `medrxiv_{year}.json`, `medrxiv.db` (Chroma), model weights |
| `output/medrxiv.db` | Chroma vector DB used by the MCP search tool |
