import requests
from bs4 import BeautifulSoup


HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"}


def retrieve_papers(start_date, end_date):
    base_url = "https://www.medrxiv.org/search/"

    query_parts = [
        "jcode%3Amedrxiv",
        "subject_collection_code%3AInfectious%20Diseases%20%28except%20HIV%252FAIDS%29",
        f"limit_from%3A{start_date}",
        f"limit_to%3A{end_date}",
        "sort%3Apublication-date",
        "direction%3Adescending",
        "format_result%3Acondensed",
        "numresults%3A75",
    ]

    url = base_url + "%20".join(query_parts)

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

    return results


def retrieve_paper_details(doi):
    server = "medrxiv"
    url = f"https://api.medrxiv.org/details/{server}/{doi}/na/json"

    response = requests.get(url, headers=HTTP_HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error fetching paper details: {response.status_code} - {response.text}")

    return response.json()["collection"][0]


def run_retrieval():
    start_date = "2025-05-01"
    end_date = "2025-05-07"

    print(f"Retrieving MedRxiv articles from {start_date} to {end_date}.")

    papers = retrieve_papers(start_date, end_date)
    for paper in papers:
        print(paper)

    paper = retrieve_paper_details(papers[0]["doi"])
    print(paper)


if __name__ == "__main__":
    run_retrieval()
