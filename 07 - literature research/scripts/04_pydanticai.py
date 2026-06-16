"""Script form of the "04 - PydanticAI" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with PydanticAI, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import asyncio

import os

import dotenv

import chromadb
from chromadb.utils import embedding_functions
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

OLLAMA_URL = "http://localhost:11434"
MAX_ROUNDS = 2

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

model = OpenAIChatModel(
    os.environ.get("GENSCAI_AGENT_MODEL", "qwen3.6:27b"),
    provider=OpenAIProvider(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"),
)

for h in research.search_medrxiv("dengue vaccination", max_results=3):
    print(h["date"], "-", h["title"][:80])

ef = embedding_functions.OllamaEmbeddingFunction(url=OLLAMA_URL, model_name="nomic-embed-text")
chroma = chromadb.PersistentClient(path=str(paths.output / "agent_frameworks" / "pydanticai_store"))
collection = chroma.get_or_create_collection("papers", embedding_function=ef)

_seen: dict[str, dict] = {}

SYSTEM = (
    "You are a research assistant building a cited literature review grounded in a curated corpus. "
    "Workflow: (1) call read_saved_papers to see what is curated; (2) use search_preprints, always "
    "including the key topic term (e.g. 'dengue') in queries; (3) for each candidate, if its abstract "
    "is plausibly relevant call save_relevant_paper(doi), erring toward saving; (4) before synthesizing "
    "you MUST call read_saved_papers and base the synthesis only on those papers, citing each by title "
    "and DOI. Never claim no papers were found if the store is non-empty. Be concise."
)

researcher = Agent(model, system_prompt=SYSTEM)


@researcher.tool_plain
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract."""
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
    blocks = []
    for doi, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        blocks.append(f"# {meta['title']}\nDOI: {doi} | Date: {meta['date']}\nURL: {meta['url']}\n\n{doc}")
    return "\n\n---\n\n".join(blocks)

result = asyncio.run(researcher.run(RESEARCH_QUESTION))
synthesis = result.output
print(synthesis)

critic = Agent(
    model,
    system_prompt=(
        "You review a literature synthesis. Reply with exactly PASS if it answers the question, cites "
        "specific papers (title + DOI), and reports evidence. Otherwise reply REVISE: <specific gap>."
    ),
)

for round_num in range(MAX_ROUNDS):
    verdict = asyncio.run(critic.run(f"Question: {RESEARCH_QUESTION}\n\nSynthesis:\n{synthesis}")).output
    print(f"--- critic (round {round_num + 1}): {verdict[:120]}")
    if "PASS" in verdict:
        break
    synthesis = asyncio.run(researcher.run(
        f"Revise your synthesis using this feedback:\n{verdict}\n\nQuestion: {RESEARCH_QUESTION}"
    )).output

print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

data = collection.get()
print(f"{len(data['ids'])} papers saved:")
for meta in data["metadatas"]:
    print(" -", meta["title"][:80])
