# genscai

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

'genscai' is a suite of generative AI tools and recipes for science and research.


## Data availability
Data is available on hugging-face (currently a private repository):
```
data = load_dataset("krosenf/midas-abstracts")
```


## Dependencies
For Hugging Face models locally (e.g. using Transformers), you'll need SentencePiece, which will be installed durring Setup. SentencePiece requires Python 3.11 or earlier, however. To use SentencePiece, you can install Python 3.11 and set up your virtual environment using 'python3.11 -m venv .venv'. Alternatively, if you're not running models locally, ignore the build error when running 'pip install'.


## Setup
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

To use developer tools such as black, install dev dependencies:
```
pip install -e '.[dev]'
```

To run tests with pytest, install test dependencies:
```
pip install -e '.[test]'
```

For editing and running the example notebooks locallly, you'll need to separately install Jupyter Lab:
```
pip install jupyterlab
```


## References
- MIDAS abstracts: [link](https://midasnetwork.us/papers/)


## Datastore
- [Private sharepoint](https://bmgf-my.sharepoint.com/:f:/g/personal/katherine_rosenfeld_gatesfoundation_org/EuwhqMcDjwpMhyFYme9FzOYBAsA4xxiuE2dOXLJtCozG8g?e=2NvSHa)
