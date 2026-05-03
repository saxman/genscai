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
jupyter lab notebooks/
```

`--all-extras` covers dev, notebooks, and CUDA groups. On Windows use `.venv\Scripts\activate`.

## Notebooks

All 23 notebooks live flat in `notebooks/`, numbered 00–09 (with x.y sub-numbers for variant series). Groups by number range: 01 Data Collection, 02–04 Text Analysis, 05–06 Knowledge & RAG, 07 Model Optimization, 08 Agents.

Notebooks 06.1 and 06.2 (RAG and RAG Evaluation) depend on `output/medrxiv.db` built by `scripts/07.2_medrxiv_knowledge_base.py`. Notebook 06.2 also requires `pip install ragas datasets`.

## Architecture

**GenScAI** is a research toolkit that pipelines literature into a searchable, agent-accessible knowledge base. The flow is:

```
MIDAS website / medRxiv API
  → scripts/01 & 07.1  (scrape / download raw JSON + TinyDB)
  → scripts/03 & 05    (prompt optimization → paper classification → modeling_papers.json)
  → scripts/07.2       (index into Chroma vector DB: output/medrxiv.db)
  → notebooks/03 - Knowledge and RAG/  (knowledge graphs, RAG, RAG evaluation)
  → genscai/tools.py   (MCP server exposing search_research_articles)
  → streamlit chatbot  (multi-LLM UI wired to MCP tools)
```

### `genscai/` package modules

| Module | Role |
|---|---|
| `paths.py` | Single source of truth for `root`, `data`, `output`, `tests` paths |
| `retrieval.py` | `MIDASRetriever` — scrapes midasnetwork.us, stores to TinyDB |
| `medrxiv.py` | medRxiv REST API client (`retrieve_articles`) |
| `classification.py` | LLM-based paper classifier; parses `[YES]`/`[NO]` responses; exports metrics |
| `training.py` | Prompt mutation templates for hill-climbing prompt optimization |
| `data.py` | Loaders for the labeled training set and MIDAS Hugging Face dataset |
| `tools.py` | FastMCP server — exposes `search_research_articles` over the MCP protocol |
| `utils.py` | `ReadOnlyTinyDB`, date extraction from paths, CUDA device info |

### Scripts pipeline (`scripts/`)

Scripts are numbered to reflect execution order:

- **01** — Retrieve MIDAS abstracts; deduplicate; split 70/15/15 into train/test/validate CSVs
- **02** — Spot-check classification quality against labeled data
- **03** — Auto-tune classification prompts via hill-climbing mutation; saves best prompts to `data/prompt_catalog.db` (TinyDB)
- **04** — Cross-model validation: test prompts tuned on one model against others
- **05** — Classify all MIDAS papers; output `data/modeling_papers.json`
- **06** — Same classification across multiple open models in parallel
- **07.1** — Download medRxiv papers for 2019–2024 to `output/medrxiv_{year}.json`
- **07.2** — Index medRxiv abstracts into Chroma (`output/medrxiv.db`); creates two collections: full abstracts and 256-char chunked with 50-char overlap
- **08** — Minimal smolagents example (web search agent)
- `papers_json_to_csv.py` — Convert classification results to modeling/non-modeling CSVs

### LLM abstraction

All LLM calls go through **`aimu`** (a thin unified client over Ollama, HuggingFace Transformers, and Anthropic). The streamlit chatbot also supports **`litellm`** for broader model routing. **`smolagents`** handles agentic workflows; **FastMCP** exposes tools over the Model Context Protocol.

### Classification prompt engineering

`classification.py` contains three prompt templates:
- `CLASSIFICATION_TASK_PROMPT_TEMPLATE` — defines the classification task
- `CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE` — enforces `[YES]`/`[NO]` output format
- Generation kwargs default to `max_new_tokens=3, temperature=0.01` for deterministic classification

`training.py` adds mutation templates for positive/negative examples used during prompt hill-climbing in `scripts/03`.

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
