"""Script form of the "05 - CrewAI" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with CrewAI, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import asyncio

import os

import dotenv

import chromadb
from chromadb.utils import embedding_functions
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

OLLAMA_URL = "http://localhost:11434"
MODEL = os.environ.get("GENSCAI_AGENT_MODEL", "qwen3.6:27b")
MAX_ROUNDS = 2

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

llm = LLM(model=f"ollama/{MODEL}", base_url=OLLAMA_URL)

for h in research.search_medrxiv("dengue vaccination", max_results=3):
    print(h["date"], "-", h["title"][:80])

ef = embedding_functions.OllamaEmbeddingFunction(url=OLLAMA_URL, model_name="nomic-embed-text")
chroma = chromadb.PersistentClient(path=str(paths.output / "agent_frameworks" / "crewai_store"))
collection = chroma.get_or_create_collection("papers", embedding_function=ef)

_seen: dict[str, dict] = {}


@tool("search_preprints")
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints for a topic. Always include the key term (e.g. 'dengue')."""
    results = research.search_medrxiv(query, max_results=5) + research.search_biorxiv(query, max_results=3)
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


@tool("save_relevant_paper")
def save_relevant_paper(doi: str) -> str:
    """Save a paper confirmed relevant to the research question, identified by its DOI."""
    article = _seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first."
    collection.upsert(
        ids=[doi],
        documents=[f"{article['title']}\n\n{article['abstract']}"],
        metadatas=[{"title": article["title"], "date": article["date"] or "", "url": article["url"] or ""}],
    )
    return f"Saved: {article['title']}"


@tool("read_saved_papers")
def read_saved_papers() -> str:
    """Read every paper saved in the local document store, to ground a synthesis."""
    data = collection.get()
    if not data["ids"]:
        return "No papers saved yet."
    blocks = []
    for doi, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
        blocks.append(f"# {meta['title']}\nDOI: {doi} | Date: {meta['date']}\nURL: {meta['url']}\n\n{doc}")
    return "\n\n---\n\n".join(blocks)

researcher = Agent(
    role="Preprint researcher",
    goal="Find and curate preprints relevant to the research question.",
    backstory="You search medRxiv and bioRxiv, judge each abstract's relevance, and save only the matches.",
    tools=[search_preprints, save_relevant_paper],
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Evidence synthesizer",
    goal="Write a concise, cited synthesis grounded only in the curated corpus.",
    backstory="You read the saved papers and write an evidence summary, citing each by title and DOI.",
    tools=[read_saved_papers],
    llm=llm,
    verbose=True,
)


def build_crew(writer_instruction):
    research_task = Task(
        description=(
            f"Research question: {RESEARCH_QUESTION}\n"
            "Search preprints (always include 'dengue' in queries). For each candidate, if its abstract "
            "is plausibly relevant, save it with save_relevant_paper. Save several relevant papers."
        ),
        expected_output="A short list of the papers you saved.",
        agent=researcher,
    )
    write_task = Task(
        description=writer_instruction,
        expected_output="A concise synthesis citing each paper by title and DOI.",
        agent=writer,
        context=[research_task],
    )
    return Crew(agents=[researcher, writer], tasks=[research_task, write_task], process=Process.sequential)


crew = build_crew(
    "Call read_saved_papers and write a synthesis that answers the research question using only those "
    "papers, citing each by title and DOI."
)
result = asyncio.run(crew.kickoff_async())
synthesis = result.raw
print("\n=== SYNTHESIS ===\n")
print(synthesis)

critic = Agent(
    role="Review editor",
    goal="Judge whether a synthesis answers the question with cited evidence.",
    backstory="You reply PASS or REVISE with specific gaps.",
    llm=llm,
    verbose=False,
)


async def critique(synthesis):
    task = Task(
        description=(
            "Reply with exactly PASS if this synthesis answers the question, cites specific papers "
            f"(title + DOI), and reports evidence. Otherwise reply REVISE: <specific gap>.\n\n"
            f"Question: {RESEARCH_QUESTION}\n\nSynthesis:\n{synthesis}"
        ),
        expected_output="PASS or REVISE: <gap>",
        agent=critic,
    )
    result = await Crew(agents=[critic], tasks=[task], process=Process.sequential).kickoff_async()
    return result.raw


for round_num in range(MAX_ROUNDS):
    verdict = asyncio.run(critique(synthesis))
    print(f"--- critic (round {round_num + 1}): {verdict[:120]}")
    if "PASS" in verdict:
        break
    revise_crew = build_crew(
        f"Revise the synthesis using this feedback: {verdict}. Call read_saved_papers and answer the "
        f"question ({RESEARCH_QUESTION}) using only the saved papers, citing each by title and DOI."
    )
    synthesis = asyncio.run(revise_crew.kickoff_async()).raw

print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

data = collection.get()
print(f"{len(data['ids'])} papers saved:")
for meta in data["metadatas"]:
    print(" -", meta["title"][:80])
