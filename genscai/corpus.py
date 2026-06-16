"""Relevance-gated research corpus: ready-made AIMU tools over a vector store.

Implements the literature-research pattern once — search an external source, let the agent
relevance-gate each hit, persist the keepers, and read the corpus back to ground a synthesis —
so a script wires it by constructing one :class:`Corpus` instead of re-deriving the three tools
and a DOI cache every time.

Built on ``aimu.memory.SemanticMemoryStore`` (vector-backed, so the corpus survives across runs
and can be retrieved by semantic relevance) and ``aimu.rag``. See the "#3" proposal in
``07 - literature research/aimu_api_sketch.md``.

Usage::

    from genscai import research, paths
    from genscai.corpus import Corpus

    corpus = Corpus(
        search_fn=lambda q: research.search_medrxiv(q, max_results=5) + research.search_biorxiv(q, max_results=3),
        persist_path=str(paths.output / "literature_research" / "aimu_corpus_store"),
        embedding_model="ollama:nomic-embed-text",
    )
    agent = Agent(client, tools=corpus.tools, ...)
"""

from __future__ import annotations

from typing import Callable, Optional

import aimu
from aimu import rag
from aimu.memory import SemanticMemoryStore

# An article is the dict returned by genscai.research.search_*: doi, title, date, url,
# abstract, authors. The corpus only relies on these keys.
Article = dict


def _format_hit(article: Article) -> str:
    """Render a search hit for the agent to judge relevance."""
    return (
        f"DOI: {article['doi']}\nTitle: {article['title']}\nDate: {article.get('date')}\n"
        f"Abstract: {(article.get('abstract') or '')[:600]}"
    )


def _format_saved(article: Article) -> str:
    """Render a saved paper for storage and read-back."""
    return (
        f"# {article['title']}\nDOI: {article['doi']} | Date: {article.get('date') or ''}\n"
        f"URL: {article.get('url') or ''}\n\n{article.get('abstract') or ''}"
    )


class Corpus:
    """A search function plus a vector store, exposed as three relevance-gated agent tools.

    ``tools`` is ``[search_preprints, save_relevant_paper, read_saved_papers]``, ready to pass
    to ``Agent(tools=...)``. The DOI cache that lets ``save_relevant_paper`` persist a paper the
    agent only named by DOI is held privately here, not in a module global.

    Args:
        search_fn: Maps a query string to a list of article dicts (e.g. wrapping
            ``genscai.research.search_medrxiv`` / ``search_biorxiv``).
        store: A prebuilt ``SemanticMemoryStore``. If ``None``, one is built from the
            ``persist_path`` / ``collection_name`` / ``embedding_model`` arguments.
        persist_path: Directory for the Chroma store; ``None`` keeps it in memory.
        collection_name: Chroma collection name.
        embedding_model: Optional ``"provider:model_id"`` for an AIMU embedding client
            (e.g. ``"ollama:nomic-embed-text"``). ``None`` uses Chroma's default embedder.
        read_top_k: Default number of papers :meth:`retrieve` returns.
    """

    def __init__(
        self,
        search_fn: Callable[[str], list[Article]],
        store: Optional[SemanticMemoryStore] = None,
        *,
        persist_path: Optional[str] = None,
        collection_name: str = "papers",
        embedding_model: Optional[str] = None,
        read_top_k: int = 8,
    ) -> None:
        if store is None:
            embedding_client = aimu.embedding_client(embedding_model) if embedding_model else None
            store = SemanticMemoryStore(
                collection_name=collection_name,
                persist_path=persist_path,
                embedding_client=embedding_client,
            )
        self.search_fn = search_fn
        self.store = store
        self.read_top_k = read_top_k
        self._seen: dict[str, Article] = {}
        self._saved_dois: set[str] = set()
        self._saved_titles: list[str] = []
        self.tools = self._build_tools()

    def _build_tools(self) -> list:
        seen = self._seen
        saved_dois = self._saved_dois
        saved_titles = self._saved_titles
        store = self.store
        search_fn = self.search_fn

        @aimu.tool
        def search_preprints(query: str) -> str:
            """Search medRxiv and bioRxiv preprints. Always include the key topic term (e.g. 'dengue')."""
            try:
                results = search_fn(query)
            except Exception as exc:
                return f"Search temporarily unavailable ({exc}). Try again shortly or rephrase the query."
            if not results:
                return "No results found."
            for article in results:
                seen[article["doi"]] = article
            return "\n\n".join(_format_hit(a) for a in results)

        @aimu.tool
        def save_relevant_paper(doi: str) -> str:
            """Save a paper you have confirmed relevant to the research question, identified by its DOI."""
            article = seen.get(doi)
            if not article:
                return f"Unknown DOI {doi}. Search for it first so its details are available."
            if doi in saved_dois:
                return f"Already saved: {article['title']}"
            store.store(_format_saved(article))
            saved_dois.add(doi)
            saved_titles.append(article["title"])
            return f"Saved: {article['title']}"

        @aimu.tool
        def read_saved_papers() -> str:
            """Read every paper saved in the local corpus, to ground your synthesis."""
            docs = store.list_all()
            if not docs:
                return "No papers saved yet."
            return "\n\n---\n\n".join(docs)

        return [search_preprints, save_relevant_paper, read_saved_papers]

    def retrieve(self, query: str, n_results: Optional[int] = None) -> list[str]:
        """Return the most semantically relevant saved papers for ``query``.

        Unlike ``read_saved_papers`` (which returns the whole corpus), this pulls only the
        top-``read_top_k`` matches — the path that keeps synthesis affordable as the corpus grows.
        """
        return rag.retrieve(self.store, query, n_results=n_results or self.read_top_k)

    @property
    def saved_titles(self) -> list[str]:
        return list(self._saved_titles)
