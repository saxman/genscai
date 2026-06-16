"""Script form of the "01 - AIMU" notebook (see ../notebooks/).

The same agentic literature-research workflow as the other notebooks in this folder, implemented with AIMU, captured as a runnable script. Requires a local Ollama server (see the folder README).

This version uses three AIMU capabilities added for this use case:
- ToolContext dependency injection, so tools reach the document store and DOI cache without module globals;
- EvaluatorOptimizer(verdict_schema=...), a typed critic verdict instead of substring-matching "PASS";
- aimu.pretty_print, which renders a streamed run (tool calls + text) without a hand-written loop.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel

import aimu
from aimu import ToolContext
from aimu.agents import Agent, EvaluatorOptimizer
from aimu.memory import DocumentStore

from genscai import paths

import shared

MODEL = f"ollama:{shared.agent_model()}"

shared.print_search_preview(show_doi=True)


@dataclass
class ResearchDeps:
    """Shared state the research tools need, injected via ToolContext instead of globals."""

    store: DocumentStore
    seen: dict[str, dict] = field(default_factory=dict)


@aimu.tool
def search_preprints(ctx: ToolContext[ResearchDeps], query: str) -> str:
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract."""
    return shared.search_preprints(query, ctx.deps.seen)


@aimu.tool
def save_relevant_paper(ctx: ToolContext[ResearchDeps], doi: str) -> str:
    """Save a paper you have confirmed is relevant to the research question, identified by its DOI."""
    article = ctx.deps.seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first so its details are available."
    content = (
        f"# {article['title']}\n\nDOI: {doi}\nAuthors: {article['authors']}\n"
        f"Date: {article['date']}\nURL: {article['url']}\n\n{article['abstract']}"
    )
    ctx.deps.store.write(f"/{doi.replace('/', '_')}.md", content)
    return f"Saved: {article['title']}"


@aimu.tool
def read_saved_papers(ctx: ToolContext[ResearchDeps]) -> str:
    """Read every paper saved in the local document store, to ground your synthesis."""
    saved = ctx.deps.store.list_paths()
    if not saved:
        return "No papers saved yet."
    return "\n\n---\n\n".join(ctx.deps.store.read(p) for p in saved)


deps = ResearchDeps(store=DocumentStore(persist_path=str(paths.output / "literature_research" / "aimu_store")))

# A longer, numbered variant of shared.RESEARCHER_SYSTEM tuned for AIMU's final-answer flow.
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
    deps=deps,
    max_iterations=12,
    reset_messages_on_run=True,
    final_answer_prompt=(
        "Call read_saved_papers, then write the final cited synthesis based only on those saved papers. "
        "Do not claim there are no papers if the store is non-empty."
    ),
    name="researcher",
)

# pretty_print renders the streamed run (tool calls + generated text); deps come from the agent's field.
aimu.pretty_print(researcher.run(shared.RESEARCH_QUESTION, stream=True))


class Verdict(BaseModel):
    passed: bool
    feedback: str = ""


# Returns a typed Verdict rather than a "PASS" string (see EvaluatorOptimizer verdict_schema below).
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
print(synthesis)

print("Saved papers:")
for path in deps.store.list_paths():
    print(" ", path)

print("\nFirst saved paper:\n")
print(deps.store.read(deps.store.list_paths()[0]))
