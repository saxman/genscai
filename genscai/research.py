"""Shared, framework-agnostic literature research tools.

These plain functions are the *only* shared code across the agent-framework comparison
notebooks ("07 - literature research/"). Each notebook adapts them into its framework's tool
abstraction, so the comparison stays focused on orchestration rather than the tools themselves.

Live sources are queried through public keyword APIs:

- Europe PMC (https://www.ebi.ac.uk/europepmc/) indexes medRxiv and bioRxiv preprints (and the
  wider literature) and exposes a clean keyword search REST API. We use it instead of scraping
  the medRxiv/bioRxiv search pages directly: bioRxiv's search is behind a Cloudflare bot
  challenge, and the rxiv detail APIs lag for the newest preprints. Europe PMC returns DOIs and
  abstracts for both servers reliably.
- arXiv via its official API (optional `arxiv` package).

Document storage is deliberately *not* here: each notebook uses its own framework's document
store (see "07 - literature research/README.md").
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
HTTP_HEADERS = {"User-Agent": "genscai-research/0.1 (https://github.com/)"}

_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2


def _get_with_retries(url, params):
    """GET with a few retries on transient errors; the public Europe PMC API occasionally 503s."""
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_error = e
            logger.warning("Europe PMC request failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error


# The comparison series targets recent preprints; callers can override the window or pass None.
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_MAX_RESULTS = 10


def _build_europepmc_query(query, src=None, publisher=None, start_date=None, end_date=None) -> str:
    """Compose a Europe PMC query string from a free-text term plus optional filters."""
    parts = [f"({query})"]
    if src:
        parts.append(f"(SRC:{src})")
    if publisher:
        parts.append(f'PUBLISHER:"{publisher}"')
    if start_date and end_date:
        parts.append(f"FIRST_PDATE:[{start_date} TO {end_date}]")
    return " AND ".join(parts)


def _normalize_europepmc(result: dict) -> dict:
    """Map a Europe PMC 'core' result into the common article shape used by all notebooks."""
    doi = result.get("doi")
    return {
        "doi": doi,
        "title": result.get("title"),
        "abstract": result.get("abstractText"),
        "authors": result.get("authorString"),
        "date": result.get("firstPublicationDate"),
        "source": result.get("publisher") or result.get("source"),
        "url": f"https://doi.org/{doi}" if doi else None,
    }


def _search_europepmc(query, max_results, *, publisher=None, start_date=None, end_date=None) -> list[dict]:
    full_query = _build_europepmc_query(
        query, src="PPR" if publisher else None, publisher=publisher, start_date=start_date, end_date=end_date
    )
    params = {
        "query": full_query,
        "format": "json",
        "resultType": "core",
        "pageSize": min(max_results, 100),
    }
    response = _get_with_retries(EUROPEPMC_SEARCH_URL, params)

    results = response.json().get("resultList", {}).get("result", [])
    return [_normalize_europepmc(r) for r in results[:max_results]]


def search_medrxiv(
    query: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict]:
    """Search medRxiv preprints (via Europe PMC) for a free-text query.

    Args:
        query: Free-text search term, e.g. "dengue vaccination".
        start_date: Earliest publication date, 'YYYY-MM-DD' (pass None to disable).
        end_date: Latest publication date, 'YYYY-MM-DD' (pass None to disable).
        max_results: Maximum number of articles to return.

    Returns:
        A list of dicts, each with: doi, title, abstract, authors, date, source, url.
    """
    return _search_europepmc(query, max_results, publisher="medRxiv", start_date=start_date, end_date=end_date)


def search_biorxiv(
    query: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict]:
    """Search bioRxiv preprints (via Europe PMC) for a free-text query.

    Args:
        query: Free-text search term.
        start_date: Earliest publication date, 'YYYY-MM-DD' (pass None to disable).
        end_date: Latest publication date, 'YYYY-MM-DD' (pass None to disable).
        max_results: Maximum number of articles to return.

    Returns:
        A list of dicts, each with: doi, title, abstract, authors, date, source, url.
    """
    return _search_europepmc(query, max_results, publisher="bioRxiv", start_date=start_date, end_date=end_date)


def search_arxiv(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict]:
    """Search arXiv for papers matching a free-text query.

    Requires the optional `arxiv` package (installed via the `agents` extra).

    Args:
        query: Free-text search term.
        max_results: Maximum number of papers to return.

    Returns:
        A list of dicts, each with: doi, title, abstract, authors, date, source, url.
    """
    import arxiv

    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results = []
    for paper in arxiv.Client().results(search):
        results.append(
            {
                "doi": paper.entry_id,
                "title": paper.title,
                "abstract": paper.summary,
                "authors": ", ".join(author.name for author in paper.authors),
                "date": paper.published.date().isoformat() if paper.published else None,
                "source": "arxiv",
                "url": paper.entry_id,
            }
        )

    return results


def fetch_article(doi: str) -> dict:
    """Fetch full details for a single article by DOI (via Europe PMC).

    Args:
        doi: The article DOI.

    Returns:
        A dict with: doi, title, abstract, authors, date, source, url. Raises ValueError if
        the DOI is not found.
    """
    params = {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 1}
    response = _get_with_retries(EUROPEPMC_SEARCH_URL, params)

    results = response.json().get("resultList", {}).get("result", [])
    if not results:
        raise ValueError(f"No article found for DOI {doi}")

    return _normalize_europepmc(results[0])
