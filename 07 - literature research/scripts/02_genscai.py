"""Script form of the "02 - genscai" notebook (see ../notebooks/).

The same agentic literature-research workflow as 01_aimu.py, but the three research tools, the DOI
cache, and the vector-backed document store all come from genscai.corpus.Corpus instead of being
hand-written. It still pairs with EvaluatorOptimizer(verdict_schema=...) for a typed critic verdict
and aimu.pretty_print for the streamed run. Requires a local Ollama server (see the folder README).
"""

from pydantic import BaseModel

import aimu
from aimu.agents import Agent, EvaluatorOptimizer

from genscai import research, paths
from genscai.corpus import Corpus

import shared

MODEL = f"ollama:{shared.agent_model()}"

shared.print_search_preview()

# One constructor wires search + relevance-gated save + read over a persistent semantic store; the
# DOI cache that lets save_relevant_paper persist a paper named only by DOI is held inside `corpus`.
corpus = Corpus(
    search_fn=lambda q: research.search_medrxiv(q, max_results=5) + research.search_biorxiv(q, max_results=3),
    persist_path=str(paths.output / "literature_research" / "aimu_corpus_store"),
    embedding_model="ollama:nomic-embed-text",
)

researcher = Agent(
    aimu.client(MODEL),
    system_message=shared.RESEARCHER_SYSTEM,
    tools=corpus.tools,
    max_iterations=12,
    reset_messages_on_run=True,
    final_answer_prompt=(
        "Call read_saved_papers, then write the final cited synthesis based only on those saved papers. "
        "Do not claim there are no papers if the store is non-empty."
    ),
    name="researcher",
)

aimu.pretty_print(researcher.run(shared.RESEARCH_QUESTION, stream=True))


class Verdict(BaseModel):
    passed: bool
    feedback: str = ""


critic = Agent(
    aimu.client(MODEL),
    system_message=shared.CRITIC_VERDICT_SYSTEM,
    max_iterations=1,
    reset_messages_on_run=True,
    name="critic",
)

review = EvaluatorOptimizer(
    generator=researcher, evaluator=critic, max_rounds=shared.MAX_ROUNDS, verdict_schema=Verdict
)

synthesis = review.run(shared.RESEARCH_QUESTION)
print("\n=== FINAL SYNTHESIS ===\n")
print(synthesis)

shared.print_saved_summary(corpus.saved_titles)
