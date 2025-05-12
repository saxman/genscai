from genscai import medxriv, paths
import chromadb
import json


def retrieve_and_store_raw_articles(year):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"Retrieving articles from {start_date} to {end_date}")
    articles = medxriv.retrieve_articles(start_date=start_date, end_date=end_date)

    print(f"Writing {len(articles)} articles to file")
    with open(paths.output / f"medxriv_{year}.json", "w") as fout:
        json.dump(articles, fout)

    return articles


def store_articles_in_chroma(articles):
    client = chromadb.PersistentClient(path=str(paths.output / "genscai.db"))

    try:
        collection = client.create_collection(name="medxriv")
    except Exception:
        collection = client.get_collection(name="medxriv")

    documents = [x["abstract"] for x in articles]
    ids = [x["doi"] for x in articles]
    metadatas = [{k: v for k, v in item.items() if k != "doi" and k != "abstract"} for item in articles]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)


if __name__ == "__main__":
    for year in range(2019, 2025):
        # articles = retrieve_and_store_raw_articles(year)

        with open(paths.output / f"medxriv_{year}.json", "r") as fin:
            articles = json.load(fin)

        print(f"Storing {len(articles)} articles for {year} in ChromaDB")
        store_articles_in_chroma(articles)
