"""Tests for genscai.corpus.Corpus.

Exercise the relevance-gated tool logic (search cache, save/dedup, read), the top-k retrieve
delegation, and the store-building factory path. A duck-typed stub stands in for
SemanticMemoryStore so no ChromaDB, embedding model, or network is touched.
"""

import genscai.corpus as corpus_mod
from genscai.corpus import Corpus, _format_hit, _format_saved


class StubStore:
    """Duck-typed SemanticMemoryStore for offline tests."""

    def __init__(self):
        self.docs = []
        self.searched = []

    def store(self, content):
        self.docs.append(content)

    def list_all(self):
        return list(self.docs)

    def search(self, query, n_results=10, max_distance=None):
        self.searched.append((query, n_results))
        return self.docs[:n_results]


ARTICLE_A = {
    "doi": "10.1/a",
    "title": "Dengue vaccine trial",
    "date": "2024-01-01",
    "url": "http://example/a",
    "abstract": "A randomized controlled trial of a dengue vaccine.",
    "authors": "A. Author",
}
ARTICLE_B = {
    "doi": "10.2/b",
    "title": "Wolbachia vector control",
    "date": "2024-02-01",
    "url": "http://example/b",
    "abstract": "Field release of Wolbachia mosquitoes.",
    "authors": "B. Author",
}


def make_corpus(results=None, store=None):
    return Corpus(search_fn=lambda q: results if results is not None else [ARTICLE_A, ARTICLE_B], store=store or StubStore())


def tools_of(corpus):
    return {t.__name__: t for t in corpus.tools}


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_tools_are_decorated_and_named():
    corpus = make_corpus()
    assert [t.__name__ for t in corpus.tools] == ["search_preprints", "save_relevant_paper", "read_saved_papers"]
    for t in corpus.tools:
        assert hasattr(t, "__tool_spec__")
        # Closures over corpus state, so no injected ToolContext parameters.
        assert t.__tool_injected__ == []
    search_props = tools_of(corpus)["search_preprints"].__tool_spec__["function"]["parameters"]["properties"]
    assert list(search_props) == ["query"]


# ---------------------------------------------------------------------------
# search_preprints
# ---------------------------------------------------------------------------


def test_search_caches_and_formats_hits():
    corpus = make_corpus()
    out = tools_of(corpus)["search_preprints"]("dengue")
    assert "10.1/a" in out and "Dengue vaccine trial" in out
    assert "10.2/b" in out
    assert set(corpus._seen) == {"10.1/a", "10.2/b"}


def test_search_no_results():
    corpus = make_corpus(results=[])
    assert tools_of(corpus)["search_preprints"]("dengue") == "No results found."


def test_search_error_is_caught():
    def boom(_q):
        raise RuntimeError("upstream 503")

    corpus = Corpus(search_fn=boom, store=StubStore())
    out = tools_of(corpus)["search_preprints"]("dengue")
    assert "Search temporarily unavailable" in out
    assert "upstream 503" in out


# ---------------------------------------------------------------------------
# save_relevant_paper
# ---------------------------------------------------------------------------


def test_save_unknown_doi_before_search():
    corpus = make_corpus()
    out = tools_of(corpus)["save_relevant_paper"]("10.9/missing")
    assert "Unknown DOI" in out
    assert corpus.store.docs == []


def test_save_after_search_persists_and_records_title():
    corpus = make_corpus()
    tools = tools_of(corpus)
    tools["search_preprints"]("dengue")
    out = tools["save_relevant_paper"]("10.1/a")
    assert out == "Saved: Dengue vaccine trial"
    assert len(corpus.store.docs) == 1
    assert "Dengue vaccine trial" in corpus.store.docs[0]
    assert corpus.saved_titles == ["Dengue vaccine trial"]


def test_save_is_idempotent_per_doi():
    corpus = make_corpus()
    tools = tools_of(corpus)
    tools["search_preprints"]("dengue")
    tools["save_relevant_paper"]("10.1/a")
    out = tools["save_relevant_paper"]("10.1/a")
    assert "Already saved" in out
    assert len(corpus.store.docs) == 1
    assert corpus.saved_titles == ["Dengue vaccine trial"]


# ---------------------------------------------------------------------------
# read_saved_papers + retrieve
# ---------------------------------------------------------------------------


def test_read_empty_then_populated():
    corpus = make_corpus()
    tools = tools_of(corpus)
    assert tools["read_saved_papers"]() == "No papers saved yet."
    tools["search_preprints"]("dengue")
    tools["save_relevant_paper"]("10.1/a")
    tools["save_relevant_paper"]("10.2/b")
    read = tools["read_saved_papers"]()
    assert "Dengue vaccine trial" in read and "Wolbachia vector control" in read
    assert "\n\n---\n\n" in read


def test_retrieve_uses_read_top_k_default():
    corpus = Corpus(search_fn=lambda q: [], store=StubStore(), read_top_k=3)
    corpus.retrieve("dengue interventions")
    assert corpus.store.searched == [("dengue interventions", 3)]


def test_retrieve_explicit_n_results_overrides_default():
    corpus = make_corpus()
    corpus.retrieve("dengue", n_results=2)
    assert corpus.store.searched[-1] == ("dengue", 2)


def test_saved_titles_returns_a_copy():
    corpus = make_corpus()
    titles = corpus.saved_titles
    titles.append("tampered")
    assert corpus.saved_titles == []


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def test_format_helpers():
    hit = _format_hit(ARTICLE_A)
    assert "DOI: 10.1/a" in hit and "Title: Dengue vaccine trial" in hit and "Abstract:" in hit
    saved = _format_saved(ARTICLE_A)
    assert saved.startswith("# Dengue vaccine trial")
    assert "DOI: 10.1/a | Date: 2024-01-01" in saved
    assert "http://example/a" in saved


def test_format_hit_truncates_long_abstract():
    article = {**ARTICLE_A, "abstract": "x" * 1000}
    assert "x" * 600 in _format_hit(article)
    assert "x" * 601 not in _format_hit(article)


# ---------------------------------------------------------------------------
# Store-building factory path (no ChromaDB / embeddings)
# ---------------------------------------------------------------------------


def test_builds_semantic_store_with_embedding_client(monkeypatch):
    captured = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(corpus_mod, "SemanticMemoryStore", FakeStore)
    monkeypatch.setattr(corpus_mod.aimu, "embedding_client", lambda model: f"emb:{model}")

    corpus = Corpus(
        search_fn=lambda q: [],
        persist_path="/tmp/store",
        collection_name="papers",
        embedding_model="ollama:nomic-embed-text",
    )
    assert isinstance(corpus.store, FakeStore)
    assert captured["kwargs"]["persist_path"] == "/tmp/store"
    assert captured["kwargs"]["collection_name"] == "papers"
    assert captured["kwargs"]["embedding_client"] == "emb:ollama:nomic-embed-text"


def test_builds_store_without_embedding_model(monkeypatch):
    captured = {}

    class FakeStore:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(corpus_mod, "SemanticMemoryStore", FakeStore)

    Corpus(search_fn=lambda q: [], persist_path=None)
    assert captured["kwargs"]["embedding_client"] is None
