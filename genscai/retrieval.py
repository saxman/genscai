import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from tinydb import TinyDB, Query

import genscai

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"
}


def retrieve_page(page_number, startdate="2024-01-01", enddate=""):
    url = "https://midasnetwork.us/wp-admin/admin-ajax.php"

    data = {
        "action": "filter_papers",
        "paged": f"{page_number}",
        "journal": "",
        "author": "",
        "title": "",
        "startdate": startdate,
        "enddate": enddate,
        "displaydefault": "",
    }

    return requests.post(url, data=data, headers=HTTP_HEADERS)


class MIDASRetriever:
    """
    Retrieves articles from the MIDAS Network website.

    Parameters
    ----------
    startdate : str
        The start date for the search query.
    enddate : str
        The end date for the search query.

    TODO: Stream to database instead of storing everything in memory.
    """

    def __init__(self, startdate="2024-01-01", enddate="", database_path=None):
        self.startdate = startdate
        self.enddate = enddate
        self.soup = None
        self.article_links = []
        self.articles = []
        self.database_path = (
            genscai.paths.data / "raw" if database_path is None else database_path
        )

        self.last_page = self._calculate_pages()

    def _calculate_pages(self):
        page = retrieve_page(1, startdate=self.startdate, enddate=self.enddate)

        self.soup = BeautifulSoup(page.text, "html.parser")
        pages = self.soup.find_all("a", {"class", "page-numbers"})

        last_page = int(pages[-2].text)

        return last_page

    def __call__(self, *args, **kwds):
        self.retrieve()

    def retrieve(self):
        print("Creating links...")
        self._create_links()
        print("Processing articles...")
        self._process_articles()
        print("Saving to database...")
        self._save_to_db()

    def _create_links(self):
        self.article_links = []
        for i in tqdm(range(1, self.last_page)):

            for article in self.soup.find_all("article"):
                link = article.find("a")
                self.article_links.append(link.get("href"))

            page = retrieve_page(i, startdate=self.startdate, enddate=self.enddate)
            self.soup = BeautifulSoup(page.text, "html.parser")

    def _process_articles(self):
        self.articles = []
        print(f"procesing {len(self.article_links)} articles")

        for link in tqdm(self.article_links):
            resp = requests.get(link, headers=HTTP_HEADERS)
            self.soup = BeautifulSoup(resp.text, "html.parser")

            data = self.soup.find_all("p", {"class", "elementor-heading-title"})

            def get_value(d, index):
                try:
                    return d[index].text
                except IndexError:
                    return "null"

            article = {
                "title": get_value(data, 0),
                "abstract": get_value(data, 1),
                "journal": get_value(data, 2),
                "reference": get_value(data, 3),
            }

            try:
                refs = data[3].find_all("a")
                article["link"] = refs[1].get("href")
            except IndexError:
                article["link"] = "null"

            self.articles.append(article)

    def _save_to_db(self):
        db = TinyDB(self.database_path / f"db_{self.startdate}_{self.enddate}.json")
        table = db.table("articles")

        for article in self.articles:
            table.insert(article)

        print(f"{len(table.all())} articles stored (total)")
