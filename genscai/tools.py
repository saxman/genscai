from genscai import paths

from fastmcp import FastMCP

import chromadb

KNOWLEDGE_BASE_PATH = str(paths.output / "medrxiv.db")
KNOWLEDGE_BASE_ID = "articles_cosign_chunked_256"

mcp = FastMCP("Genscai MCP Server")


@mcp.tool
def hello(name: str) -> str:
    return f"Hello, {name}!"


@mcp.tool()
def search_research_articles(search_request: str) -> str:
    """
    Search for current research articles about infectious diseases and disease modeling per a given search request.

    Args:
        search_request: The information that the user is looking for.

    Returns:
        Current research articles on infectious diseases and disease modeling for the given topic.
    """

    client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_PATH)
    collection = client.get_collection(name=KNOWLEDGE_BASE_ID)
    results = collection.query(query_texts=[search_request], n_results=10)

    ids = [x for x in results["ids"][0]]
    abstracts = [x for x in results["documents"][0]]
    metadata = [x for x in results["metadatas"][0]]

    # Each article has multiple chunks, and each chunk id is the article's doi with an index appended to it.
    # To get a unique set of articles, we need to remove the index from the id and keep only one copy of article doi.
    articles = []
    content = "Relevant research articles:\n\n"
    id_set = set()
    for i in range(len(ids)):
        id = ids[i].split(":")[0]

        if id in id_set:
            continue

        id_set.add(id)
        articles.append({"id": id, "abstract": abstracts[i], "metadata": metadata[i]})
        content += f"Title: {metadata[i]['title']}\nAbstract: {abstracts[i]}\nAuthors: {metadata[i]['authors']}\nDate: {metadata[i]['date']}\nLink: https://www.medrxiv.org/content/{id}\n\n"

    return content


if __name__ == "__main__":
    mcp.run()