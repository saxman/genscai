"""Script form of the "03 - LangGraph" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with LangGraph, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import os
from typing import TypedDict

import dotenv

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

MODEL = os.environ.get("GENSCAI_AGENT_MODEL", "qwen3.6:27b")
MAX_ROUNDS = 2

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

model = ChatOllama(model=MODEL)

for h in research.search_medrxiv("dengue vaccination", max_results=3):
    print(h["date"], "-", h["title"][:80])

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(
    collection_name="papers",
    embedding_function=embeddings,
    persist_directory=str(paths.output / "agent_frameworks" / "langgraph_store"),
)

_seen: dict[str, dict] = {}


@tool
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract.

    Always include the key topic term (e.g. 'dengue') in the query.
    """
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


@tool
def save_relevant_paper(doi: str) -> str:
    """Save a paper you have confirmed relevant to the research question, identified by its DOI."""
    article = _seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first."
    doc = Document(
        page_content=f"{article['title']}\n\n{article['abstract']}",
        metadata={"doi": doi, "title": article["title"], "date": article["date"] or "", "url": article["url"] or ""},
    )
    store.add_documents([doc], ids=[doi])
    return f"Saved: {article['title']}"


@tool
def read_saved_papers() -> str:
    """Read every paper saved in the local document store, to ground your synthesis."""
    data = store.get()
    if not data["ids"]:
        return "No papers saved yet."
    blocks = []
    for meta, doc in zip(data["metadatas"], data["documents"]):
        blocks.append(f"# {meta['title']}\nDOI: {meta['doi']} | Date: {meta['date']}\nURL: {meta['url']}\n\n{doc}")
    return "\n\n---\n\n".join(blocks)

SYSTEM = (
    "You are a research assistant building a cited literature review grounded in a curated corpus. "
    "Workflow: (1) call read_saved_papers to see what is curated; (2) use search_preprints, always "
    "including the key topic term (e.g. 'dengue') in queries; (3) for each candidate, if its abstract "
    "is plausibly relevant call save_relevant_paper(doi), erring toward saving; (4) before synthesizing "
    "you MUST call read_saved_papers and base the synthesis only on those papers, citing each by title "
    "and DOI. Never claim no papers were found if the store is non-empty. Be concise."
)

researcher = create_agent(model, [search_preprints, save_relevant_paper, read_saved_papers], system_prompt=SYSTEM)


def run_researcher(task):
    result = researcher.invoke({"messages": [("user", task)]})
    return result["messages"][-1].content

CRITIC = (
    "You review a literature synthesis. Reply with exactly PASS if it answers the question, cites "
    "specific papers (title + DOI), and reports evidence. Otherwise reply REVISE: <specific gap>."
)


class State(TypedDict):
    question: str
    synthesis: str
    feedback: str
    rounds: int


def research_node(state: State) -> dict:
    if state["feedback"]:
        task = f"Revise your synthesis using this feedback:\n{state['feedback']}\n\nQuestion: {state['question']}"
    else:
        task = state["question"]
    return {"synthesis": run_researcher(task), "rounds": state["rounds"] + 1}


def evaluate_node(state: State) -> dict:
    feedback = model.invoke(
        [("system", CRITIC), ("user", f"Question: {state['question']}\n\nSynthesis:\n{state['synthesis']}")]
    ).content
    print(f"--- critic (round {state['rounds']}): {feedback[:120]}")
    return {"feedback": feedback}


def route(state: State) -> str:
    if "PASS" in state["feedback"] or state["rounds"] >= MAX_ROUNDS:
        return END
    return "research"


graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("evaluate", evaluate_node)
graph.add_edge(START, "research")
graph.add_edge("research", "evaluate")
graph.add_conditional_edges("evaluate", route, {"research": "research", END: END})
app = graph.compile()

final = app.invoke({"question": RESEARCH_QUESTION, "synthesis": "", "feedback": "", "rounds": 0})
print("\n=== FINAL SYNTHESIS ===\n")
print(final["synthesis"])

data = store.get()
print(f"{len(data['ids'])} papers saved:")
for meta in data["metadatas"]:
    print(" -", meta["title"][:80])
