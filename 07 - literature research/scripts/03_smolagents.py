"""Script form of the "03 - smolagents" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with smolagents, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import chromadb
from chromadb.utils import embedding_functions
from smolagents import LiteLLMModel, ToolCallingAgent, tool

from genscai import paths

import shared

MODEL = f"ollama_chat/{shared.agent_model()}"

model = LiteLLMModel(model_id=MODEL)

shared.print_search_preview()

ef = embedding_functions.OllamaEmbeddingFunction(url=shared.OLLAMA_URL, model_name="nomic-embed-text")
chroma = chromadb.PersistentClient(path=str(paths.output / "agent_frameworks" / "smolagents_store"))
collection = chroma.get_or_create_collection("papers", embedding_function=ef)

# Cache of search hits so a paper can be saved by DOI alone.
_seen: dict[str, dict] = {}


@tool
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract.

    Args:
        query: A free-text search query; always include the key topic term (e.g. 'dengue').
    """
    return shared.search_preprints(query, _seen)


@tool
def save_relevant_paper(doi: str) -> str:
    """Save a paper you have confirmed is relevant to the research question, identified by its DOI.

    Args:
        doi: The DOI of a paper returned by a previous search.
    """
    article = _seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first so its details are available."
    document = f"{article['title']}\n\n{article['abstract']}"
    collection.upsert(
        ids=[doi],
        documents=[document],
        metadatas=[
            {
                "title": article["title"],
                "authors": article["authors"] or "",
                "date": article["date"] or "",
                "url": article["url"] or "",
            }
        ],
    )
    return f"Saved: {article['title']}"


@tool
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


agent = ToolCallingAgent(
    tools=[search_preprints, save_relevant_paper, read_saved_papers],
    model=model,
    instructions=shared.RESEARCHER_SYSTEM,
    max_steps=12,
)

synthesis = agent.run(shared.RESEARCH_QUESTION)
print(synthesis)


def critique(question, synthesis):
    prompt = f"{shared.CRITIC_SYSTEM}\n\nQuestion: {question}\n\nSynthesis:\n{synthesis}"
    response = model.generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    return response.content


for round_num in range(shared.MAX_ROUNDS):
    feedback = critique(shared.RESEARCH_QUESTION, synthesis)
    print(f"--- critic (round {round_num + 1}): {feedback[:120]}")
    if "PASS" in feedback:
        break
    synthesis = agent.run(
        f"Revise your synthesis using this feedback:\n{feedback}\n\nQuestion: {shared.RESEARCH_QUESTION}"
    )

print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

shared.print_saved_summary([meta["title"] for meta in collection.get()["metadatas"]])
