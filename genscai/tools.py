from genscai import paths

from fastmcp import FastMCP
from fastmcp import Client

import chromadb
import asyncio

KNOWLEDGE_BASE_PATH = str(paths.output / "medrxiv.db")
KNOWLEDGE_BASE_ID = "articles_cosign_chunked_256"

MCP_CONFIG = {
    "mcpServers": {
        # "weather": {"url": "https://weather-api.example.com/mcp"},
        "assistant": {"command": "python", "args": ["../mcp/server.py"]}
    }
}

mcp = FastMCP("Genscai MCP Server")


@mcp.tool()
def search_research_articles(search_request: str) -> str:
    """
    Search for current research articles in infectious diseases and disease modeling per a given search request.

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


class MCPClient:
    def __init__(self, config: dict = None):
        self.config = config
        self.loop = asyncio.new_event_loop()

    async def _call_tool(self, tool_name: str, params: dict):
        if self.config:
            async with Client(self.config) as client:
                return await client.call_tool(tool_name, params)
        else:
            async with Client(mcp) as client:
                return await client.call_tool(tool_name, params)

    def call_tool(self, tool_name: str, params: dict):
        return self.loop.run_until_complete(self._call_tool(tool_name, params))

    async def _list_tools(self):
        if self.config:
            async with Client(self.config) as client:
                return await client.list_tools()
        else:
            async with Client(mcp) as client:
                return await client.list_tools()

    def list_tools(self):
        return self.loop.run_until_complete(self._list_tools())
