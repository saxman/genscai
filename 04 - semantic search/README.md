# Semantic Search

Index medRxiv abstracts into a Chroma vector store and run and evaluate retrieval-augmented generation (RAG) over it using embedding-based semantic search.

Part of the [genscai](../README.md) use-case series. Shared library code lives in the [`genscai`](../genscai) package; datasets in [`data/`](../data) and generated artifacts in `output/` are shared at the repo root.

## Notebooks

| # | Notebook |
| --- | --- |
| 01 | [01 - RAG - Pipeline](notebooks/01%20-%20RAG%20-%20Pipeline.ipynb) |
| 02 | [02 - RAG - Evaluation](notebooks/02%20-%20RAG%20-%20Evaluation.ipynb) |

## Scripts

- [`01_medrxiv_download.py`](scripts/01_medrxiv_download.py)
- [`02_medrxiv_knowledge_base.py`](scripts/02_medrxiv_knowledge_base.py)

## Notes

The scripts build `output/medrxiv.db`, the shared semantic-search index also consumed by the `06 - agents` and `09 - evaluation` use cases and queried by the `search_research_articles` MCP tool. Run `scripts/01_medrxiv_download.py` then `scripts/02_medrxiv_knowledge_base.py` (both thin CLIs over `genscai.knowledge_base`, which holds the importable, tested download/index logic). The RAG Evaluation notebook also needs `pip install ragas datasets`.
