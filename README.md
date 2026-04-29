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

Start with [00 - Setup.ipynb](notebooks/00%20-%20Setup.ipynb) and work through the numbered notebooks in order.

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

| Notebook | Topic |
| --- | --- |
| 00 | Environment setup |
| 01 | Information retrieval |
| 02.1 to 02.3 | Information extraction (OpenAI, LangChain, LangExtract) |
| 03.1 to 03.4 | Embeddings (general, SPECTER, SPECTER2, NV-Embed) |
| 04 | Clustering |
| 05.1 to 05.2 | Classification and evaluation |
| 06.1 to 06.2 | Knowledge graphs (GraphRAG, LangChain) |
| 07 | Model Context Protocol (MCP) |
| 08 | Model quantization with Hugging Face |
| 09 | Agents with smolagents |
| 99.1 | Article download utilities |

## Development

Run linting and tests:

```bash
ruff check .
pytest
```

## License

Released under the terms of the [LICENSE](LICENSE) file in this repository.

## References

- MIDAS Network paper abstracts: <https://midasnetwork.us/papers/>
- LASER disease modeling framework: <https://github.com/InstituteforDiseaseModeling/laser>
- Starsim disease modeling framework: <https://github.com/starsimhub/starsim>
