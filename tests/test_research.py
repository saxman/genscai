"""Tests for the shared literature research tools (genscai.research).

These exercise the pure query-building and result-normalization logic without network access.
The live search functions wrap these around an HTTP call to Europe PMC.
"""

from genscai.research import _build_europepmc_query, _normalize_europepmc


def test_query_includes_src_publisher_and_date():
    q = _build_europepmc_query(
        "dengue vaccination",
        src="PPR",
        publisher="medRxiv",
        start_date="2024-01-01",
        end_date="2025-12-31",
    )
    assert q == '(dengue vaccination) AND (SRC:PPR) AND PUBLISHER:"medRxiv" AND FIRST_PDATE:[2024-01-01 TO 2025-12-31]'


def test_query_omits_optional_filters():
    assert _build_europepmc_query("malaria") == "(malaria)"


def test_query_omits_date_when_only_one_bound_given():
    q = _build_europepmc_query("malaria", src="PPR", start_date="2024-01-01", end_date=None)
    assert "FIRST_PDATE" not in q
    assert q == "(malaria) AND (SRC:PPR)"


def test_normalize_maps_core_fields_and_builds_doi_url():
    result = {
        "doi": "10.1101/2024.04.19.24306097",
        "title": "A dengue study",
        "abstractText": "We model dengue transmission.",
        "authorString": "Smith J, Doe A.",
        "firstPublicationDate": "2024-04-21",
        "publisher": "medRxiv",
        "source": "PPR",
    }
    normalized = _normalize_europepmc(result)

    assert normalized == {
        "doi": "10.1101/2024.04.19.24306097",
        "title": "A dengue study",
        "abstract": "We model dengue transmission.",
        "authors": "Smith J, Doe A.",
        "date": "2024-04-21",
        "source": "medRxiv",
        "url": "https://doi.org/10.1101/2024.04.19.24306097",
    }


def test_normalize_handles_missing_doi():
    normalized = _normalize_europepmc({"title": "No DOI here", "source": "PPR"})
    assert normalized["doi"] is None
    assert normalized["url"] is None
    assert normalized["source"] == "PPR"
