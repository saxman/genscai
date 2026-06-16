"""Script form of the "04 - LangGraph" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with LangGraph, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

from genscai import paths

import shared

model = ChatOllama(model=shared.agent_model())

shared.print_search_preview()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(
    collection_name="papers",
    embedding_function=embeddings,
    persist_directory=str(paths.output / "agent_frameworks" / "langgraph_store"),
)

# Cache of search hits so a paper can be saved by DOI alone.
_seen: dict[str, dict] = {}


@tool
def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract.

    Always include the key topic term (e.g. 'dengue') in the query.
    """
    return shared.search_preprints(query, _seen)


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
    blocks = [
        shared.format_saved_paper(meta["title"], meta["doi"], meta["date"], meta["url"], doc)
        for meta, doc in zip(data["metadatas"], data["documents"])
    ]
    return "\n\n---\n\n".join(blocks)


researcher = create_agent(
    model, [search_preprints, save_relevant_paper, read_saved_papers], system_prompt=shared.RESEARCHER_SYSTEM
)


def run_researcher(task):
    result = researcher.invoke({"messages": [("user", task)]})
    return result["messages"][-1].content


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
        [
            ("system", shared.CRITIC_SYSTEM),
            ("user", f"Question: {state['question']}\n\nSynthesis:\n{state['synthesis']}"),
        ]
    ).content
    print(f"--- critic (round {state['rounds']}): {feedback[:120]}")
    return {"feedback": feedback}


def route(state: State) -> str:
    if "PASS" in state["feedback"] or state["rounds"] >= shared.MAX_ROUNDS:
        return END
    return "research"


graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("evaluate", evaluate_node)
graph.add_edge(START, "research")
graph.add_edge("research", "evaluate")
graph.add_conditional_edges("evaluate", route, {"research": "research", END: END})
app = graph.compile()

final = app.invoke({"question": shared.RESEARCH_QUESTION, "synthesis": "", "feedback": "", "rounds": 0})
print("\n=== FINAL SYNTHESIS ===\n")
print(final["synthesis"])

shared.print_saved_summary([meta["title"] for meta in store.get()["metadatas"]])
