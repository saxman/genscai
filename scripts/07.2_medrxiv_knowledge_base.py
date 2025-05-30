from genscai import paths
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import json
from tqdm import tqdm


def store_articles(articles):
    client = chromadb.PersistentClient(path=str(paths.output / "genscai.db"))

    collection = client.get_or_create_collection(name="medxriv")

    documents = [x["abstract"] for x in articles]
    ids = [x["doi"] for x in articles]
    metadatas = [{k: v for k, v in item.items() if k != "doi" and k != "abstract"} for item in articles]

    collection.add(documents=documents, ids=ids, metadatas=metadatas)


def store_article_chunks(articles):
    client = chromadb.PersistentClient(path=str(paths.output / "genscai.db"))

    collection = client.get_or_create_collection(name="medrxiv_chunked_256_cosine", metadata={"hnsw:space": "cosine"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

    for article in tqdm(articles):
        chunks = text_splitter.split_text(article["abstract"])
        ids = [f"{article['doi']}:{i}" for i in range(len(chunks))]
        metadatas = [{k: v for k, v in article.items() if k != "doi" and k != "abstract"} for _ in chunks]

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)


if __name__ == "__main__":
    for year in range(2019, 2025):
        with open(paths.output / f"medrxiv_{year}.json", "r") as fin:
            articles = json.load(fin)

        print(f"Storing {len(articles)} articles for year {year} in Chroma database")

        # store_articles(articles)
        store_article_chunks(articles)
