# Text Analysis

Analyze the collected literature: extract structured information, embed and cluster paper abstracts, and classify papers with local LLMs. The classification scripts also tune prompts and produce the labeled `modeling_papers.json` outputs.

Part of the [genscai](../README.md) use-case series. Shared library code lives in the [`genscai`](../genscai) package; datasets in [`data/`](../data) and generated artifacts in `output/` are shared at the repo root.

## Notebooks

| # | Notebook |
| --- | --- |
| 01 | [01 - Information Extraction - OpenAI](notebooks/01%20-%20Information%20Extraction%20-%20OpenAI.ipynb) |
| 02 | [02 - Information Extraction - LangChain](notebooks/02%20-%20Information%20Extraction%20-%20LangChain.ipynb) |
| 03 | [03 - Information Extraction - LangExtract](notebooks/03%20-%20Information%20Extraction%20-%20LangExtract.ipynb) |
| 04 | [04 - Embeddings - Vector Search](notebooks/04%20-%20Embeddings%20-%20Vector%20Search.ipynb) |
| 05 | [05 - Embeddings - SPECTER](notebooks/05%20-%20Embeddings%20-%20SPECTER.ipynb) |
| 06 | [06 - Embeddings - SPECTER2](notebooks/06%20-%20Embeddings%20-%20SPECTER2.ipynb) |
| 07 | [07 - Embeddings - NV-Embed](notebooks/07%20-%20Embeddings%20-%20NV-Embed.ipynb) |
| 08 | [08 - Embeddings - Clustering and Visualization](notebooks/08%20-%20Embeddings%20-%20Clustering%20and%20Visualization.ipynb) |
| 09 | [09 - Classification - Local LLMs](notebooks/09%20-%20Classification%20-%20Local%20LLMs.ipynb) |
| 10 | [10 - Classification - Prompt Evaluation](notebooks/10%20-%20Classification%20-%20Prompt%20Evaluation.ipynb) |

## Scripts

- [`01_classification_test.py`](scripts/01_classification_test.py)
- [`02_classification_training.py`](scripts/02_classification_training.py)
- [`03_classification_cross_model_validation.py`](scripts/03_classification_cross_model_validation.py)
- [`04_classification.py`](scripts/04_classification.py)
- [`05_classification_all_models.py`](scripts/05_classification_all_models.py)
- [`papers_json_to_csv.py`](scripts/papers_json_to_csv.py)
