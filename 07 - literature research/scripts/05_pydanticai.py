"""Script form of the "05 - PydanticAI" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with PydanticAI, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import asyncio

import chromadb
from chromadb.utils import embedding_functions
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genscai import paths

import shared

model = OpenAIChatModel(
    shared.agent_model(),
    provider=OpenAIProvider(base_url=f"{shared.OLLAMA_URL}/v1", api_key="ollama"),
)

shared.print_search_preview()

ef = embedding_functions.OllamaEmbeddingFunction(url=shared.OLLAMA_URL, model_name="nomic-embed-text")
chroma = chromadb.PersistentClient(path=str(paths.output / "agent_frameworks" / "pydanticai_store"))
collection = chroma.get_or_create_collection("papers", embedding_function=ef)

# Cache of search hits so a paper can be saved by DOI alone.
_seen: dict[str, dict] = {}

researcher = Agent(model, system_prompt=shared.RESEARCHER_SYSTEM)


@researcher.tool_plain
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract."""
    return shared.search_preprints(query, _seen)


@researcher.tool_plain
def save_relevant_paper(doi: str) -> str:
    """Save a paper you have confirmed relevant to the research question, identified by its DOI."""
    article = _seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first."
    collection.upsert(
        ids=[doi],
        documents=[f"{article['title']}\n\n{article['abstract']}"],
        metadatas=[{"title": article["title"], "date": article["date"] or "", "url": article["url"] or ""}],
    )
    return f"Saved: {article['title']}"


@researcher.tool_plain
def read_saved_papers() -> str:
    """Read every paper saved in the local document store, to ground your synthesis."""
    data = collection.get()
    if not data["ids"]:
        return "No papers saved yet."
    blocks = [
        shared.format_saved_paper(meta["title"], doi, meta["date"], meta["url"], doc)
        for doi, meta, doc in zip(data["ids"], data["metadatas"], data["documents"])
    ]
    return "\n\n---\n\n".join(blocks)


result = asyncio.run(researcher.run(shared.RESEARCH_QUESTION))
synthesis = result.output
print(synthesis)

critic = Agent(model, system_prompt=shared.CRITIC_SYSTEM)

for round_num in range(shared.MAX_ROUNDS):
    verdict = asyncio.run(critic.run(f"Question: {shared.RESEARCH_QUESTION}\n\nSynthesis:\n{synthesis}")).output
    print(f"--- critic (round {round_num + 1}): {verdict[:120]}")
    if "PASS" in verdict:
        break
    synthesis = asyncio.run(
        researcher.run(f"Revise your synthesis using this feedback:\n{verdict}\n\nQuestion: {shared.RESEARCH_QUESTION}")
    ).output

print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

shared.print_saved_summary([meta["title"] for meta in collection.get()["metadatas"]])
