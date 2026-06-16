"""Build and manage the medRxiv Chroma knowledge base.

The knowledge base (`output/medrxiv.db`) is a cross-cutting prerequisite: it is queried by the MCP
`search_research_articles` tool (`genscai.tools`) and consumed by the Knowledge & RAG, Agents, and
Evaluation use cases. This module holds the download and indexing logic so it is importable and
testable; the `04 - semantic search/scripts/` scripts are thin CLI wrappers over these functions.
"""

import json
from pathlib import Path

import chromadb

from genscai import medrxiv, paths

DB_PATH = str(paths.output / "medrxiv.db")
ABSTRACTS_COLLECTION = "articles"
# Note: "cosign" is a historical typo preserved so existing databases and genscai.tools keep working.
CHUNKED_COLLECTION = "articles_cosign_chunked_256"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50


def download_articles(years, output_dir=None) -> int:
    """Download medRxiv articles for each year to ``output/medrxiv_{year}.json``.

    Args:
        years: Iterable of years (e.g. ``range(2019, 2025)``).
        output_dir: Directory for the per-year JSON files. Defaults to ``paths.output``.

    Returns:
        Total number of articles written across all years.
    """
    output_dir = Path(output_dir) if output_dir else paths.output
    total = 0
    for year in years:
        articles = medrxiv.retrieve_articles(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
        with open(output_dir / f"medrxiv_{year}.json", "w") as fout:
            json.dump(articles, fout)
        total += len(articles)
    return total


def _chunk_article(article: dict, splitter) -> tuple[list, list, list]:
    """Split one article's abstract into chunks with stable ids and per-chunk metadata.

    Each chunk id is the article DOI suffixed with the chunk index, so all chunks of a paper can be
    deduplicated back to one DOI at query time.
    """
    chunks = splitter.split_text(article["abstract"])
    ids = [f"{article['doi']}:{i}" for i in range(len(chunks))]
    metadatas = [{k: v for k, v in article.items() if k != "doi"} for _ in chunks]
    return ids, chunks, metadatas


def store_articles(articles, db_path: str = DB_PATH) -> int:
    """Index full abstracts (one document per article) into the abstracts collection."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=ABSTRACTS_COLLECTION)
    collection.add(
        documents=[a["abstract"] for a in articles],
        ids=[a["doi"] for a in articles],
        metadatas=[{k: v for k, v in a.items() if k not in ("doi", "abstract")} for a in articles],
    )
    return collection.count()


def store_article_chunks(
    articles, db_path: str = DB_PATH, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> int:
    """Index chunked abstracts (cosine space) into the chunked collection used by the search tool."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=CHUNKED_COLLECTION, metadata={"hnsw:space": "cosine"})
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for article in articles:
        ids, chunks, metadatas = _chunk_article(article, splitter)
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return collection.count()


def build_knowledge_base(years, *, chunked: bool = True, db_path: str = DB_PATH, source_dir=None) -> int:
    """Index downloaded per-year medRxiv JSON files into the Chroma knowledge base.

    Args:
        years: Iterable of years whose ``medrxiv_{year}.json`` files to index.
        chunked: Index 256-char chunks (the format the search tool expects) when True, else full abstracts.
        db_path: Chroma database path. Defaults to ``output/medrxiv.db``.
        source_dir: Directory holding the per-year JSON files. Defaults to ``paths.output``.

    Returns:
        The collection's total item count after indexing.
    """
    source_dir = Path(source_dir) if source_dir else paths.output
    count = 0
    for year in years:
        with open(source_dir / f"medrxiv_{year}.json") as fin:
            articles = json.load(fin)
        count = (
            store_article_chunks(articles, db_path=db_path) if chunked else store_articles(articles, db_path=db_path)
        )
    return count
