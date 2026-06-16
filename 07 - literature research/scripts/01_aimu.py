"""Script form of the "01 - AIMU" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with AIMU, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import os

import dotenv

import aimu
from aimu.agents import Agent, EvaluatorOptimizer
from aimu.memory import DocumentStore

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

MODEL = f"ollama:{os.environ.get('GENSCAI_AGENT_MODEL', 'qwen3.6:27b')}"

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

hits = research.search_medrxiv("dengue vaccination", max_results=3)
for h in hits:
    print(h["date"], "-", h["title"][:80])
    print("   ", h["doi"])

store = DocumentStore(persist_path=str(paths.output / "literature_research" / "aimu_store"))

# Cache of search hits so a paper can be saved by DOI alone.
_seen: dict[str, dict] = {}


@aimu.tool
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract."""
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


@aimu.tool
def save_relevant_paper(doi: str) -> str:
    """Save a paper you have confirmed is relevant to the research question, identified by its DOI."""
    article = _seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first so its details are available."
    content = (
        f"# {article['title']}\n\nDOI: {doi}\nAuthors: {article['authors']}\n"
        f"Date: {article['date']}\nURL: {article['url']}\n\n{article['abstract']}"
    )
    store.write(f"/{doi.replace('/', '_')}.md", content)
    return f"Saved: {article['title']}"


@aimu.tool
def read_saved_papers() -> str:
    """Read every paper saved in the local document store, to ground your synthesis."""
    saved = store.list_paths()
    if not saved:
        return "No papers saved yet."
    return "\n\n---\n\n".join(store.read(p) for p in saved)

researcher = Agent(
    aimu.client(MODEL),
    system_message=(
        "You are a research assistant building a cited literature review grounded in a curated corpus.\n"
        "Workflow:\n"
        "1. Call read_saved_papers first to see what is already curated.\n"
        "2. Use search_preprints to find candidate papers. Always include the key topic term from the "
        "question (e.g. 'dengue') in every search query so results stay on-topic.\n"
        "3. For EACH candidate, judge whether its abstract is relevant to the question. If it is plausibly "
        "relevant, call save_relevant_paper(doi); err toward saving rather than discarding. Skip only "
        "clearly off-topic hits.\n"
        "4. Before writing any synthesis you MUST call read_saved_papers again, and base the synthesis "
        "ONLY on the papers it returns, citing each by title and DOI. The store already contains relevant "
        "papers, so never claim no papers were found. Be concise."
    ),
    tools=[search_preprints, save_relevant_paper, read_saved_papers],
    max_iterations=12,
    reset_messages_on_run=True,
    final_answer_prompt=(
        "Call read_saved_papers, then write the final cited synthesis based only on those saved papers. "
        "Do not claim there are no papers if the store is non-empty."
    ),
    name="researcher",
)


def stream_run(runner, task):
    """Print an AIMU agent/workflow run, marking tool calls and streaming text."""
    for chunk in runner.run(task, stream=True):
        if chunk.is_tool_call():
            print(f"\n  [tool] {chunk.content}")
        elif chunk.is_text():
            print(chunk.content, end="")
    print()


stream_run(researcher, RESEARCH_QUESTION)

critic = Agent(
    aimu.client(MODEL),
    system_message=(
        "You review a literature synthesis for an infectious-disease researcher. The synthesis is built "
        "from a curated corpus of saved preprints. Reply with exactly PASS if it answers the question, "
        "cites specific papers (title + DOI), and reports evidence from them. Only reply REVISE: <specific "
        "gap> when something concrete is missing. Do NOT ask for topics outside the saved corpus, and do "
        "not penalize a focused answer. Prefer PASS when the synthesis is reasonable."
    ),
    max_iterations=1,
    reset_messages_on_run=True,
    name="critic",
)

review = EvaluatorOptimizer(generator=researcher, evaluator=critic, max_rounds=2, pass_keyword="PASS")

synthesis = review.run(RESEARCH_QUESTION)
print(synthesis)

print("Saved papers:")
for path in store.list_paths():
    print(" ", path)

print("\nFirst saved paper:\n")
print(store.read(store.list_paths()[0]))
