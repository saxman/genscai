# Data Collection

Retrieve research-paper abstracts and articles from the sources the rest of the project builds on — the MIDAS network, medRxiv/arXiv, and Paperclip — and split them into train/test/validate sets.

Part of the [genscai](../README.md) use-case series. Shared library code lives in the [`genscai`](../genscai) package; datasets in [`data/`](../data) and generated artifacts in `output/` are shared at the repo root.

## Notebooks

| # | Notebook |
| --- | --- |
| 01 | [01 - Data Collection - Abstracts from MIDAS](notebooks/01%20-%20Data%20Collection%20-%20Abstracts%20from%20MIDAS.ipynb) |
| 02 | [02 - Data Collection - Articles from medRxiv and arXiv](notebooks/02%20-%20Data%20Collection%20-%20Articles%20from%20medRxiv%20and%20arXiv.ipynb) |
| 03 | [03 - Data Collection - Paperclip](notebooks/03%20-%20Data%20Collection%20-%20Paperclip.ipynb) |

## Scripts

- [`01_retrieve_information.py`](scripts/01_retrieve_information.py)
