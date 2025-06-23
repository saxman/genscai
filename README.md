# GenScAI

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/saxman/genscai)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Fgenscai%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

'GenScAI' is a suite of generative AI tools and recipes for science and research. The project includes Python modules for research and science tasks, such as downloading and processing research articles. It also includes notebooks demonstrating how to use LLMs for science and research tasks, and an example chatbot [notebooks/genscai_chatbot.py] for augmenting sceintific workflows.

## Requirements

### Python
Python 3.10+

### Hugging Face
For using LLMs from Hugging Face (e.g. using Transformers), you'll need the [SentencePiece module](https://pypi.org/project/sentencepiece/), which will be installed during project setup (below). SentencePiece requires Python 3.11 or earlier, however. To use SentencePiece, you can install Python 3.11 and set up your virtual environment using 'python3.11 -m venv .venv' (or 'uv venv --python 3.11' if using uv). Alternatively, if you don't intend to run Hugging Face models locally, ignore the build error when running 'pip install' during setup.

### Ollama
For locally using LLMs via Ollama, you'll need to download and install Ollama from https://ollama.com/.

## Setup

### With [uv](https://docs.astral.sh/uv/) [recommended]

Set up a virtual Python environment. The '--seed' argument has uv add pip and setuptools to the virtual environment. This is optional, and is only useful if using an IDE that accesses pip directly (e.g. Visual Studio Code). 
```
uv venv --seed
```

Activate (enter) the virtual environment:
```
source .venv/bin/activate
```

Install all dependencies, including development and notebook dependencies:
```
uv sync --all-extras
```

Alternatively, dependencies can be installed per use case.

Install main project dependencies:
```
uv pip install -e .
```

To use developer tools such as ruff and pytest, install the 'dev' dependencies:
```
uv pip install -e '.[dev]'
```

For editing and running the example notebooks, you'll want to install the following supporting modules:
```
uv pip install -e '.[notebooks]'
```

And finally, for running LLMs locally on systems with nvidia GPUs:
```
uv pip install -e '.[cuda]'
```

### Without uv

Set up a virtual Python environment:
```
python3 -m venv .venv
```

Activate (enter) the virtual environment:
```
source .venv/bin/activate
```

Build the genscai module and install dependencies:
```
pip install -e .
```

To use developer tools such as ruff and pytest, install the 'dev' dependencies:
```
pip install -e '.[dev]'
```

For editing and running the example notebooks, you'll want to install the following supporting modules:
```
pip install -e '.[notebooks]'
```

And finally, for running LLMs locally on systems with nvidia GPUs:
```
pip install -e '.[cuda]'
```

## References
- MIDAS Papaer Abstracts: [link](https://midasnetwork.us/papers/)
