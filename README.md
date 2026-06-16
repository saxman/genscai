# GenScAI

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/saxman/genscai)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Fgenscai%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

**Generative AI tools and recipes for science and research.**

GenScAI is a hands-on toolkit for applying large language models to scientific workflows, with a focus on infectious disease research and modeling. It pairs a reusable Python package with runnable notebooks and example applications so researchers can retrieve literature, extract structured information, classify and cluster papers, build knowledge graphs, and prototype LLM agents using local or hosted models.

## Features

- **Literature retrieval** from sources such as medRxiv and configurable APIs.
- **Information extraction** recipes using OpenAI, LangChain, and LangExtract.
- **Embeddings and clustering** with general-purpose and science-specific models (SPECTER, SPECTER2, NV-Embed).
- **Document classification** with training, evaluation, and cross-model validation pipelines.
- **Knowledge graph construction** via GraphRAG and LangChain.
- **Agents and tools** built on `smolagents` and the Model Context Protocol (MCP).
- **Relevance-gated research corpus** ([`genscai.corpus.Corpus`](genscai/corpus.py)): search → relevance-gate → persist → read as ready-made AIMU agent tools over a vector store.
- **Local model support** through Ollama, Hugging Face Transformers, and quantization recipes.
- **Example chatbot** ([streamlit/genscai_chatbot.py](streamlit/genscai_chatbot.py)) for exploring infectious disease modeling frameworks (LASER, Starsim) and the research literature.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) (optional, for running LLMs locally)
- An NVIDIA GPU and CUDA drivers (optional, for accelerated local inference)

## Installation

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

`--all-extras` installs the main, development, notebook, and CUDA dependency groups. To install only the dependencies you need:

```bash
uv pip install -e .                 # core package
uv pip install -e '.[dev]'          # ruff, pytest
uv pip install -e '.[notebooks]'    # ipykernel, ipywidgets
uv pip install -e '.[cuda]'         # NVIDIA GPU support
```

### With pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                    # add '[dev]', '[notebooks]', or '[cuda]' as needed
```

## Quick start

Launch the example chatbot:

```bash
streamlit run streamlit/genscai_chatbot.py
```

Open the notebooks (each use-case folder has a `notebooks/` and/or `scripts/` subdir):

```bash
jupyter lab
```

New here? Start with [00 - getting started/notebooks/01 - Environment Setup.ipynb](00%20-%20getting%20started/notebooks/01%20-%20Environment%20Setup.ipynb).

## Project layout

Each scientific use case is a top-level folder containing its own `notebooks/` and/or `scripts/`. The
reusable Python package, datasets, generated artifacts, and tests are shared at the root.

```
genscai/
├── genscai/                   Reusable Python package (retrieval, classification, research, corpus, knowledge_base, simulation, tools)
├── 00 - getting started/      Environment setup
├── 01 - data collection/      Retrieve abstracts/articles (MIDAS, medRxiv, arXiv)
├── 02 - text analysis/        Information extraction, embeddings, classification
├── 03 - knowledge graphs/     Knowledge-graph construction and querying (GraphRAG, LangChain)
├── 04 - semantic search/      medRxiv vector index + retrieval-augmented generation (RAG)
├── 05 - model optimization/   Quantization and fine-tuning
├── 06 - agents/               Agent + MCP tool demos
├── 07 - literature research/  Agentic literature research across six frameworks
├── 08 - disease simulation/   Compartmental-model intervention-planning agent
├── 09 - evaluation/           Benchmarking
├── streamlit/                 Example Streamlit applications (the GenScAI chatbot)
├── data/                      Sample and training datasets
├── output/                    Generated artifacts (vector DB, model weights)
├── tests/                     Package tests
└── pyproject.toml
```

## Use cases

Each top-level folder is a use case with its own README listing its notebooks and scripts.

| Use case | What it covers |
| --- | --- |
| [00 - getting started](00%20-%20getting%20started/) | Environment setup |
| [01 - data collection](01%20-%20data%20collection/) | Retrieve abstracts/articles (MIDAS, medRxiv, arXiv) |
| [02 - text analysis](02%20-%20text%20analysis/) | Information extraction, embeddings, classification |
| [03 - knowledge graphs](03%20-%20knowledge%20graphs/) | Knowledge-graph construction and querying (GraphRAG, LangChain) |
| [04 - semantic search](04%20-%20semantic%20search/) | medRxiv vector index and retrieval-augmented generation (RAG) |
| [05 - model optimization](05%20-%20model%20optimization/) | Quantization and fine-tuning |
| [06 - agents](06%20-%20agents/) | Tool-using agents (MCP, smolagents, AIMU workflows) |
| [07 - literature research](07%20-%20literature%20research/) | Agentic literature research across six frameworks + a `genscai.corpus` build |
| [08 - disease simulation](08%20-%20disease%20simulation/) | Compartmental-model intervention-planning agent |
| [09 - evaluation](09%20-%20evaluation/) | Benchmarking models and agent configurations |

Within a folder, notebooks are numbered from `01` and meant to be worked through in order.

### Literature research (agent frameworks)

The [`07 - literature research/`](07%20-%20literature%20research/) folder implements one identical use case — agentic
literature research over medRxiv/bioRxiv with a relevance-gated local document store and a critic
feedback loop — **six ways across agent frameworks** plus a "batteries-included" build from the
project's own [`genscai.corpus.Corpus`](genscai/corpus.py) helper (on AIMU), all on local Ollama
models. Read them side by side. Requires `uv sync --all-extras`. See its
[README](07%20-%20literature%20research/README.md).

| # | Implementation |
| --- | --- |
| 01 | [AIMU](07%20-%20literature%20research/notebooks/01%20-%20AIMU.ipynb) |
| 02 | [genscai](07%20-%20literature%20research/notebooks/02%20-%20genscai.ipynb) (`genscai.corpus.Corpus`, on AIMU) |
| 03 | [smolagents](07%20-%20literature%20research/notebooks/03%20-%20smolagents.ipynb) |
| 04 | [LangGraph](07%20-%20literature%20research/notebooks/04%20-%20LangGraph.ipynb) |
| 05 | [PydanticAI](07%20-%20literature%20research/notebooks/05%20-%20PydanticAI.ipynb) |
| 06 | [CrewAI](07%20-%20literature%20research/notebooks/06%20-%20CrewAI.ipynb) |
| 07 | [LlamaIndex](07%20-%20literature%20research/notebooks/07%20-%20LlamaIndex.ipynb) |

### Disease simulation

The [`08 - disease simulation/`](08%20-%20disease%20simulation/) folder holds an adaptive intervention-planning agent
([scripts/01_intervention_agent.py](08%20-%20disease%20simulation/scripts/01_intervention_agent.py)) that drives
the compartmental model in `genscai/simulation.py`.

## Development

Run linting and tests:

```bash
ruff check .
pytest
```

## License

Released under the [Apache License 2.0](LICENSE).

## References

- MIDAS Network paper abstracts: <https://midasnetwork.us/papers/>
- LASER disease modeling framework: <https://github.com/InstituteforDiseaseModeling/laser>
- Starsim disease modeling framework: <https://github.com/starsimhub/starsim>
