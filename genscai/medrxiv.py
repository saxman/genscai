import requests
import urllib.parse
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"}


def retrieve_article_ids(start_date, end_date) -> list:
    """
    Retrieve a list of article DOIs from medRxiv within a specified date range.

    Args:
        start_date (str): The start date for the search in the format 'YYYY-MM-DD'.
        end_date (str): The end date for the search in the format 'YYYY-MM-DD'.

    Returns:
        list: A list of dictionaries containing article DOIs referenced as 'doi'.
    """
    articles = []
    next_page = 0

    while True:
        a, next_page = _retrieve_article_ids_by_page(start_date, end_date, next_page)
        articles.extend(a)
        if next_page is None:
            break

    return articles


def _retrieve_article_ids_by_page(start_date="2025-01-01", end_date="2025-01-31", page=0) -> tuple:
    """
    Retrieve a single page of article DOIs from medRxiv within a specified date range.

    Args:
        start_date (str): The start date for the search in the format 'YYYY-MM-DD'.
        end_date (str): The end date for the search in the format 'YYYY-MM-DD'.
        page (int): The page number to retrieve.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries with article DOIs referenced as 'doi'.
            - int or None: The next page number, or None if there are no more pages.
    """
    base_url = "https://www.medrxiv.org/search/"

    query_parts = [
        "jcode:medrxiv",
        "subject_collection_code:Infectious Diseases (except HIV%2FAIDS)",
        f"limit_from:{start_date}",
        f"limit_to:{end_date}",
        "sort:publication-date",
        "direction:descending",
        "format_result:condensed",
        "numresults:75",  # can be up to 75
    ]

    url = base_url + urllib.parse.quote(" ".join(query_parts)) + f"?page={page}"
    response = requests.post(url, headers=HTTP_HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error fetching list of articles: {response.status_code} - {response.text}")

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("li", class_="search-result")

    results = []
    for article in articles:
        result = {
            "doi": "/".join(article.find("span", class_="highwire-cite-metadata-doi").text.strip().split("/")[-2:]),
            # "title": article.find('span', class_='highwire-cite-title').text.strip(),
            # "authors": article.find('span', class_='highwire-citation-authors').text.strip(),
        }

        results.append(result)

    next_page = None
    pager_element = soup.find("ul", class_="pager-items")

    if pager_element:
        last_pager_item_anchor_element = pager_element.find("li", class_="last").find("a")

        if last_pager_item_anchor_element:
            next_page = page + 1

    return (results, next_page)


def retrieve_article_details(doi) -> dict:
    """
    Retrieve detailed information about an article from medRxiv using its DOI.

    Args:
        doi (str): The DOI of the article.

    Returns:
        dict: A dictionary containing detailed information about the article.
    """
    server = "medrxiv"
    url = f"https://api.medrxiv.org/details/{server}/{doi}/na/json"

    response = requests.get(url, headers=HTTP_HEADERS)

    if response.status_code != 200:
        raise Exception(
            f"Error fetching article details for DOI {doi}: Invalid HTTP response: {response.status_code} - {response.text}"
        )

    if response.json()["collection"] is None:
        raise Exception(f"Error fetching article details for DOI {doi}: No collection element in JSON response")

    if len(response.json()["collection"]) == 0:
        raise Exception(f"Error fetching article details for DOI {doi}: No article details in JSON response")

    return response.json()["collection"][0]


def retrieve_articles(start_date, end_date) -> dict:
    """
    Retrieve detailed information about articles from medRxiv within a specified date range.

    Args:
        start_date (str): The start date for the search in the format 'YYYY-MM-DD'.
        end_date (str): The end date for the search in the format 'YYYY-MM-DD'.

    Returns:
        dict: A dictionary containing detailed information about the articles.
    """
    articles = retrieve_article_ids(start_date, end_date)
    results = []

    for article in articles:
        try:
            result = retrieve_article_details(article["doi"])
            results.append(result)
        except Exception as e:
            logger.error(f"Error retrieving article details for DOI {article['doi']}: {e}")
            continue

    return results


def __test_retrieval():
    """
    Run the retrieval process for articles within a specific date range.

    This function retrieves article DOIs for a predefined date range, prints the DOIs,
    and fetches detailed information for the first article.
    """
    start_date = "2025-01-01"
    end_date = "2025-01-31"

    articles = retrieve_article_ids(start_date, end_date)

    for article in articles:
        print(article)

    article = retrieve_article_details(articles[0]["doi"])
    print(article)


if __name__ == "__main__":
    __test_retrieval()
