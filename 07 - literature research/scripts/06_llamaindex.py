"""Script form of the "06 - LlamaIndex" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with LlamaIndex, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import asyncio

import os

import dotenv

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Workflow, step, Event, StartEvent, StopEvent
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

MODEL = os.environ.get("GENSCAI_AGENT_MODEL", "qwen3.6:27b")
OLLAMA_URL = "http://localhost:11434"
MAX_ROUNDS = 2

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

llm = Ollama(model=MODEL, base_url=OLLAMA_URL, request_timeout=600.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=OLLAMA_URL)

for h in research.search_medrxiv("dengue vaccination", max_results=3):
    print(h["date"], "-", h["title"][:80])

index = VectorStoreIndex.from_documents([], embed_model=embed_model)

_seen: dict[str, dict] = {}


def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Always include the key topic term (e.g. 'dengue')."""
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


def save_relevant_paper(doi: str) -> str:
    """Save a paper confirmed relevant to the research question, identified by its DOI."""
    article = _seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first."
    index.insert(
        Document(
            text=f"{article['title']}\n\n{article['abstract']}",
            metadata={"doi": doi, "title": article["title"], "date": article["date"] or "", "url": article["url"] or ""},
            doc_id=doi,
        )
    )
    return f"Saved: {article['title']}"

class SynthesizeEvent(Event):
    feedback: str = ""


class EvaluateEvent(Event):
    synthesis: str


RESEARCHER_SYSTEM = (
    "You curate a corpus of preprints. Use search_preprints (always include 'dengue' in queries). "
    "For each candidate, if its abstract is plausibly relevant, call save_relevant_paper(doi), erring "
    "toward saving. Save 4 to 6 relevant papers, then reply 'DONE' and stop. Do not keep searching "
    "after you have saved enough."
)


class ResearchWorkflow(Workflow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rounds = 0

    @step
    async def research(self, ev: StartEvent) -> SynthesizeEvent:
        # ReActAgent uses text-based ReAct prompting, which works with local Ollama models that
        # lack reliable native tool-calling (FunctionAgent's tool schema trips some model templates).
        agent = ReActAgent(
            tools=[search_preprints, save_relevant_paper], llm=llm, system_prompt=RESEARCHER_SYSTEM
        )
        # A local model may keep looping past a sensible stopping point and exhaust the agent's step
        # budget; the relevant papers are already saved to the index, so we proceed to synthesis either way.
        try:
            await agent.run(RESEARCH_QUESTION)
        except Exception as exc:
            print(f"(research agent stopped early: {exc})")
        return SynthesizeEvent()

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> EvaluateEvent:
        self.rounds += 1
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=8)
        instruction = (
            "Write a concise synthesis answering: "
            f"{RESEARCH_QUESTION} Cite each paper by title and DOI from the context."
        )
        if ev.feedback:
            instruction += f"\nImprove using this feedback: {ev.feedback}"
        response = await query_engine.aquery(instruction)
        return EvaluateEvent(synthesis=str(response))

    @step
    async def evaluate(self, ev: EvaluateEvent) -> SynthesizeEvent | StopEvent:
        verdict = str(
            await llm.acomplete(
                "Reply with exactly PASS if this synthesis answers the question with cited evidence "
                f"(title + DOI). Otherwise reply REVISE: <specific gap>.\n\n"
                f"Question: {RESEARCH_QUESTION}\n\nSynthesis:\n{ev.synthesis}"
            )
        )
        print(f"--- critic (round {self.rounds}): {verdict[:120]}")
        if "PASS" in verdict or self.rounds >= MAX_ROUNDS:
            return StopEvent(result=ev.synthesis)
        return SynthesizeEvent(feedback=verdict)

workflow = ResearchWorkflow(timeout=1200, verbose=True)
synthesis = asyncio.run(workflow.run())
print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

docs = index.docstore.docs
print(f"{len(docs)} papers saved:")
for node in docs.values():
    print(" -", node.metadata.get("title", "")[:80])
