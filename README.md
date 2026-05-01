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

Open a notebook to explore a specific recipe:

```bash
jupyter lab notebooks/
```

Start with [00 - Environment Setup.ipynb](notebooks/00%20-%20Environment%20Setup.ipynb), then open the group folders in `notebooks/` and work through them in numbered order.

## Project layout

```
genscai/
├── genscai/        Reusable Python package (retrieval, classification, training, tools)
├── notebooks/      Numbered, runnable recipes covering retrieval through agents
├── scripts/        Standalone scripts for batch retrieval, training, and classification
├── streamlit/      Example Streamlit applications (e.g. the GenScAI chatbot)
├── data/           Sample and training datasets (MIDAS abstracts, modeling papers)
└── pyproject.toml
```

## Notebooks

Start with [00 - Environment Setup.ipynb](notebooks/00%20-%20Environment%20Setup.ipynb) and work through the numbered notebooks in order.

| # | Title | Group |
| --- | --- | --- |
| 00 | [Environment Setup](notebooks/00%20-%20Environment%20Setup.ipynb) | — |
| 01 | [Abstract Retrieval from MIDAS](notebooks/01%20-%20Abstract%20Retrieval%20from%20MIDAS.ipynb) | Data Collection |
| 02 | [Article Download from medRxiv and arXiv](notebooks/02%20-%20Article%20Download%20from%20medRxiv%20and%20arXiv.ipynb) | Data Collection |
| 03.1 | [Information Extraction - OpenAI](notebooks/03.1%20-%20Information%20Extraction%20-%20OpenAI.ipynb) | Text Analysis |
| 03.2 | [Information Extraction - LangChain](notebooks/03.2%20-%20Information%20Extraction%20-%20LangChain.ipynb) | Text Analysis |
| 03.3 | [Information Extraction - LangExtract](notebooks/03.3%20-%20Information%20Extraction%20-%20LangExtract.ipynb) | Text Analysis |
| 04.1 | [Embeddings - Chroma](notebooks/04.1%20-%20Embeddings%20-%20Chroma.ipynb) | Text Analysis |
| 04.2 | [Embeddings - SPECTER](notebooks/04.2%20-%20Embeddings%20-%20SPECTER.ipynb) | Text Analysis |
| 04.3 | [Embeddings - SPECTER2](notebooks/04.3%20-%20Embeddings%20-%20SPECTER2.ipynb) | Text Analysis |
| 04.4 | [Embeddings - NV-Embed](notebooks/04.4%20-%20Embeddings%20-%20NV-Embed.ipynb) | Text Analysis |
| 05 | [Embedding Clustering and Visualization](notebooks/05%20-%20Embedding%20Clustering%20and%20Visualization.ipynb) | Text Analysis |
| 06.1 | [Paper Classification with Open LLMs](notebooks/06.1%20-%20Paper%20Classification%20with%20Open%20LLMs.ipynb) | Text Analysis |
| 06.2 | [Classification Prompt Evaluation](notebooks/06.2%20-%20Classification%20Prompt%20Evaluation.ipynb) | Text Analysis |
| 07.1 | [Knowledge Graph - GraphRAG](notebooks/07.1%20-%20Knowledge%20Graph%20-%20GraphRAG.ipynb) | Knowledge & RAG |
| 07.2 | [Knowledge Graph - LangChain](notebooks/07.2%20-%20Knowledge%20Graph%20-%20LangChain.ipynb) | Knowledge & RAG |
| 08 | [Retrieval-Augmented Generation](notebooks/08%20-%20Retrieval-Augmented%20Generation.ipynb) | Knowledge & RAG |
| 09 | [RAG Evaluation](notebooks/09%20-%20RAG%20Evaluation.ipynb) | Knowledge & RAG |
| 10 | [Model Quantization with Hugging Face](notebooks/10%20-%20Model%20Quantization%20with%20Hugging%20Face.ipynb) | Local Models |
| 11 | [Model Fine-Tuning with Hugging Face](notebooks/11%20-%20Model%20Fine-Tuning%20with%20Hugging%20Face.ipynb) | Local Models |
| 12 | [Agentic Tools with MCP](notebooks/12%20-%20Agentic%20Tools%20with%20MCP.ipynb) | Agents |
| 13 | [AI Agents with smolagents](notebooks/13%20-%20AI%20Agents%20with%20smolagents.ipynb) | Agents |

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
