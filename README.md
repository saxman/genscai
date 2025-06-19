# genscai

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/saxman/genscai)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsaxman%2Fgenscai%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)

  
'genscai' is a suite of generative AI tools and recipes for science and research.


## Data availability
Data is available on hugging-face (currently a private repository):
```
data = load_dataset("krosenf/midas-abstracts")
```


## Requirements

Python 3.10+

For using Hugging Face models locally (e.g. using Transformers), you'll need SentencePiece, which will be installed during project setup (below). SentencePiece requires Python 3.11 or earlier, however. To use SentencePiece, you can install Python 3.11 and set up your virtual environment using 'python3.11 -m venv .venv' (or 'uv venv --python 3.11' if using uv). Alternatively, if you don't intend to run Hugging Face models locally, ignore the build error when running 'pip install' during setup.


## Setup (without uv)

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

To use developer tools such as black and pytest, install dev dependencies:
```
pip install -e '.[dev]'
```

For editing and running the example notebooks locally, you'll need to separately install Jupyter Lab:
```
pip install jupyterlab
```

## Setup (with uv)
Set up a virtual Python environment. The '--seed' argument has uv add pip and setuptools to the virtual environment. This is optional, and is only useful if using an IDE that accesses pip directly (e.g. Visual Studio Code). 
```
uv venv --seed
```

Activate (enter) the virtual environment:
```
source .venv/bin/activate
```

Build the genscai module and install dependencies:
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

## References
- MIDAS abstracts: [link](https://midasnetwork.us/papers/)


## Datastore
- [Private sharepoint](https://bmgf-my.sharepoint.com/:f:/g/personal/katherine_rosenfeld_gatesfoundation_org/EuwhqMcDjwpMhyFYme9FzOYBAsA4xxiuE2dOXLJtCozG8g?e=2NvSHa)
