"""Script form of the "07 - LlamaIndex" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with LlamaIndex, captured as a runnable script. Requires a local Ollama server (see the folder README).
"""

import asyncio

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Workflow, step, Event, StartEvent, StopEvent
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

import shared

llm = Ollama(model=shared.agent_model(), base_url=shared.OLLAMA_URL, request_timeout=600.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url=shared.OLLAMA_URL)

shared.print_search_preview()

index = VectorStoreIndex.from_documents([], embed_model=embed_model)

# Cache of search hits so a paper can be saved by DOI alone.
_seen: dict[str, dict] = {}


def search_preprints(query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Always include the key topic term (e.g. 'dengue')."""
    return shared.search_preprints(query, _seen)


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


# A ReAct-specific variant of shared.RESEARCHER_SYSTEM: it tells the agent to save a handful of
# papers and then stop, since the synthesis runs as a separate step.
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
            await agent.run(shared.RESEARCH_QUESTION)
        except Exception as exc:
            print(f"(research agent stopped early: {exc})")
        return SynthesizeEvent()

    @step
    async def synthesize(self, ev: SynthesizeEvent) -> EvaluateEvent:
        self.rounds += 1
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=8)
        instruction = (
            "Write a concise synthesis answering: "
            f"{shared.RESEARCH_QUESTION} Cite each paper by title and DOI from the context."
        )
        if ev.feedback:
            instruction += f"\nImprove using this feedback: {ev.feedback}"
        response = await query_engine.aquery(instruction)
        return EvaluateEvent(synthesis=str(response))

    @step
    async def evaluate(self, ev: EvaluateEvent) -> SynthesizeEvent | StopEvent:
        verdict = str(
            await llm.acomplete(
                f"{shared.CRITIC_SYSTEM}\n\nQuestion: {shared.RESEARCH_QUESTION}\n\nSynthesis:\n{ev.synthesis}"
            )
        )
        print(f"--- critic (round {self.rounds}): {verdict[:120]}")
        if "PASS" in verdict or self.rounds >= shared.MAX_ROUNDS:
            return StopEvent(result=ev.synthesis)
        return SynthesizeEvent(feedback=verdict)


workflow = ResearchWorkflow(timeout=1200, verbose=True)
synthesis = asyncio.run(workflow.run())
print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

docs = index.docstore.docs
shared.print_saved_summary([node.metadata.get("title", "") for node in docs.values()])
