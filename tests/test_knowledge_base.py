"""Tests for genscai.knowledge_base.

Covers the pure chunking logic and the download loop (whose earlier script version was mis-indented
and only ever fetched the final year). No network or Chroma access required.
"""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from genscai import knowledge_base


def test_chunk_article_ids_and_metadata():
    article = {
        "doi": "10.1101/2024.01.01.24300001",
        "abstract": "word " * 200,  # long enough to split into several chunks
        "title": "A study",
        "date": "2024-01-02",
    }
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    ids, chunks, metadatas = knowledge_base._chunk_article(article, splitter)

    assert len(ids) == len(chunks) == len(metadatas) > 1
    # ids are the DOI suffixed with the chunk index, so chunks dedupe back to one paper
    assert ids == [f"{article['doi']}:{i}" for i in range(len(chunks))]
    # doi is excluded from metadata (it lives in the id); other fields are retained
    assert all("doi" not in m for m in metadatas)
    assert all(m["title"] == "A study" for m in metadatas)


def test_download_articles_writes_every_year(monkeypatch, tmp_path):
    calls = []

    def fake_retrieve_articles(start_date, end_date):
        calls.append((start_date, end_date))
        return [{"doi": f"d-{start_date}", "abstract": "a"}]

    monkeypatch.setattr(knowledge_base.medrxiv, "retrieve_articles", fake_retrieve_articles)

    years = [2019, 2020, 2021]
    total = knowledge_base.download_articles(years, output_dir=tmp_path)

    # One retrieval per year (the old script fetched only the last year), one file per year
    assert [c[0] for c in calls] == ["2019-01-01", "2020-01-01", "2021-01-01"]
    assert total == len(years)
    for year in years:
        path = tmp_path / f"medrxiv_{year}.json"
        assert path.exists()
        assert json.loads(path.read_text())[0]["doi"] == f"d-{year}-01-01"
