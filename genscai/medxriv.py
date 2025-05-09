import requests
import urllib.parse
from bs4 import BeautifulSoup


HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"}


def retrieve_articles(start_date, end_date):
    articles = []
    next_page = 0

    while True:
        a, next_page = retrieve_articles(start_date, end_date, next_page)
        articles.extend(a)
        if next_page is None:
            break

    return articles


def retrieve_articles_by_page(start_date, end_date, page=0):
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
        raise Exception(f"Error fetching paper details: {response.status_code} - {response.text}")

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("li", class_="search-result")

    results = []
    for article in articles:
        result = {
            # "title": article.find('span', class_='highwire-cite-title').text.strip(),
            # "authors": article.find('span', class_='highwire-citation-authors').text.strip(),
            "doi": "/".join(article.find("span", class_="highwire-cite-metadata-doi").text.strip().split("/")[-2:])
        }

        results.append(result)

    next_page = None
    pager_element = soup.find("ul", class_="pager-items")

    # if there are multiple pages...
    if pager_element:
        last_pager_item_anchor_element = pager_element.find("li", class_="last").find("a")

        # if the element of the last page includes an anchor, we're not on the last page yet
        if last_pager_item_anchor_element:
            next_page = page + 1

    return (results, next_page)


def retrieve_article_details(doi):
    server = "medrxiv"
    url = f"https://api.medrxiv.org/details/{server}/{doi}/na/json"

    response = requests.get(url, headers=HTTP_HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error fetching paper details: {response.status_code} - {response.text}")

    return response.json()["collection"][0]


def run_retrieval():
    start_date = "2025-05-01"
    end_date = "2025-05-07"

    articles = retrieve_articles(start_date, end_date)

    for article in articles:
        print(article)

    # paper = retrieve_article_details(papers[0]["doi"])
    # print(paper)


if __name__ == "__main__":
    run_retrieval()
