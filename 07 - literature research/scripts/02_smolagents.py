"""Script form of the "02 - smolagents" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with smolagents, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import os

import dotenv

import chromadb
from chromadb.utils import embedding_functions
from smolagents import LiteLLMModel, ToolCallingAgent, tool

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

MODEL = f"ollama_chat/{os.environ.get('GENSCAI_AGENT_MODEL', 'qwen3.6:27b')}"
OLLAMA_URL = "http://localhost:11434"

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

model = LiteLLMModel(model_id=MODEL)

for h in research.search_medrxiv("dengue vaccination", max_results=3):
    print(h["date"], "-", h["title"][:80])

ef = embedding_functions.OllamaEmbeddingFunction(url=OLLAMA_URL, model_name="nomic-embed-text")
chroma = chromadb.PersistentClient(path=str(paths.output / "agent_frameworks" / "smolagents_store"))
collection = chroma.get_or_create_collection("papers", embedding_function=ef)

_seen: dict[str, dict] = {}


@tool
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract.

    Args:
        query: A free-text search query; always include the key topic term (e.g. 'dengue').
    """
    try:
        results = research.search_medrxiv(query, max_results=5) + research.search_biorxiv(query, max_results=3)
    except Exception as exc:
        return f"Search temporarily unavailable ({exc}). Try again shortly or rephrase the query."
    if not results:
        return "No results found."
    blocks = []
    for article in results:
        _seen[article["doi"]] = article
        blocks.append(
            f"DOI: {article['doi']}\nTitle: {article['title']}\nDate: {article['date']}\n"
            f"Abstract: {(article['abstract'] or '')[:600]}"
        )
    return "\n\n".join(blocks)


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
        metadatas=[{"title": article["title"], "authors": article["authors"] or "", "date": article["date"] or "", "url": article["url"] or ""}],
    )
    return f"Saved: {article['title']}"


@tool
def read_saved_papers() -> str:
    """Read every paper saved in the local document store, to ground your synthesis."""
    data = collection.get()
    if not data["ids"]:
        return "No papers saved yet."
    blocks = []
    for doi, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        blocks.append(f"# {meta['title']}\nDOI: {doi} | Date: {meta['date']}\nURL: {meta['url']}\n\n{doc}")
    return "\n\n---\n\n".join(blocks)

SYSTEM = (
    "You are a research assistant building a cited literature review grounded in a curated corpus. "
    "Workflow: (1) call read_saved_papers to see what is curated; (2) use search_preprints, always "
    "including the key topic term (e.g. 'dengue') in queries; (3) for each candidate, if its abstract "
    "is plausibly relevant call save_relevant_paper(doi), erring toward saving; (4) before synthesizing "
    "you MUST call read_saved_papers and base the synthesis only on those papers, citing each by title "
    "and DOI. Never claim no papers were found if the store is non-empty. Be concise."
)

agent = ToolCallingAgent(
    tools=[search_preprints, save_relevant_paper, read_saved_papers],
    model=model,
    instructions=SYSTEM,
    max_steps=12,
)

synthesis = agent.run(RESEARCH_QUESTION)
print(synthesis)

def critique(question, synthesis):
    prompt = (
        "Review this literature synthesis. Reply with exactly PASS if it answers the question, cites "
        "specific papers (title + DOI), and reports evidence. Otherwise reply REVISE: <specific gap>.\n\n"
        f"Question: {question}\n\nSynthesis:\n{synthesis}"
    )
    response = model.generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    return response.content


MAX_ROUNDS = 2
for round_num in range(MAX_ROUNDS):
    feedback = critique(RESEARCH_QUESTION, synthesis)
    print(f"--- critic (round {round_num + 1}): {feedback[:120]}")
    if "PASS" in feedback:
        break
    synthesis = agent.run(f"Revise your synthesis using this feedback:\n{feedback}\n\nQuestion: {RESEARCH_QUESTION}")

print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

data = collection.get()
print(f"{len(data['ids'])} papers saved:")
for meta in data["metadatas"]:
    print(" -", meta["title"][:80])
