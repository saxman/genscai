# AIMU API sketch — validating the proposed improvements

> **This is a design sketch, not runnable code.** The APIs marked `# PROPOSED` do not exist in
> AIMU today. The point is to see whether the four proposed improvements actually reduce the
> friction in [`scripts/01_aimu.py`](scripts/01_aimu.py) before anyone changes the framework.
> See the analysis in `~/.claude/plans/how-could-the-aimu-tranquil-moth.md`.

The four proposals being tested:
1. **Tool context injection** — a schema-invisible `ctx` param the agent fills from `run(..., deps=)`.
2. **Structured/predicate verdict** — `EvaluatorOptimizer(verdict_schema=...)` instead of `pass_keyword="PASS"`.
3. **Research-corpus helper** — `aimu.tools.research_corpus(...)` returning wired search/save/read tools.
4. **Streaming print helper** — `aimu.pretty_print(...)` instead of a hand-written chunk loop.

---

## Version A — high-level, using the research-corpus helper (#3 + #2 + #4)

This is the version that proposal #3 makes possible. The three tools, the `_seen` cache, and the
`stream_run` helper all disappear. **Current script: 117 lines. This sketch: ~40.**

```python
"""Script form of the "01 - AIMU" notebook, against the proposed AIMU API."""

from pydantic import BaseModel

import aimu
from aimu.agents import Agent, EvaluatorOptimizer
from aimu.memory import SemanticMemoryStore
from aimu.tools import research_corpus  # PROPOSED (#3)

from genscai import research, paths

import shared

MODEL = f"ollama:{shared.agent_model()}"

# PROPOSED (#3): one factory wires search + relevance-gated save + top-k read over a vector store.
# The DOI cache that the tools share is held inside `corpus`, not a module global.
corpus = research_corpus(
    search_fn=lambda q: research.search_medrxiv(q, max_results=5) + research.search_biorxiv(q, max_results=3),
    store=SemanticMemoryStore(
        collection_name="papers",
        persist_path=str(paths.output / "literature_research" / "aimu_store"),
        embedding_client=aimu.embedding_client("ollama:nomic-embed-text"),
    ),
    read_top_k=8,  # synthesis pulls the 8 most relevant saved papers, not the whole corpus
)

researcher = Agent(
    aimu.client(MODEL),
    system_message=shared.RESEARCHER_SYSTEM,
    tools=corpus.tools,            # PROPOSED (#3): [search_preprints, save_relevant_paper, read_saved_papers]
    max_iterations=12,
    reset_messages_on_run=True,
    final_answer_prompt="Call read_saved_papers, then write the final cited synthesis from those papers only.",
    name="researcher",
)


class Verdict(BaseModel):          # PROPOSED (#2)
    passed: bool
    feedback: str = ""


critic = Agent(
    aimu.client(MODEL),
    system_message=shared.CRITIC_SYSTEM,   # no more "reply with EXACTLY PASS" — output is typed
    max_iterations=1,
    reset_messages_on_run=True,
    name="critic",
)

# PROPOSED (#2): loop stops on verdict.passed, not a substring grep.
review = EvaluatorOptimizer(generator=researcher, evaluator=critic,
                            max_rounds=shared.MAX_ROUNDS, verdict_schema=Verdict)

aimu.pretty_print(review.run(shared.RESEARCH_QUESTION, stream=True))  # PROPOSED (#4)
```

---

## Version B — low-level, keeping hand-written tools (#1 + #2 + #4)

If you *don't* want the corpus helper (say the persistence backend is custom), proposal #1 alone
still removes the module globals. The tools take an injected `ctx`; state lives on `deps`.

```python
from dataclasses import dataclass, field

import aimu
from aimu.agents import Agent
from aimu.memory import DocumentStore


@dataclass
class ResearchDeps:                # the shared state, passed explicitly instead of via globals
    store: DocumentStore
    seen: dict[str, dict] = field(default_factory=dict)


@aimu.tool
def search_preprints(ctx: aimu.ToolContext[ResearchDeps], query: str) -> str:  # PROPOSED (#1)
    """Search medRxiv and bioRxiv preprints. Returns each hit's DOI, title, date, and abstract."""
    # `ctx` is injected by the agent and stripped from the tool's JSON schema, so the model
    # still only sees `query`. No module-global `_seen`.
    return shared.search_preprints(query, ctx.deps.seen)


@aimu.tool
def save_relevant_paper(ctx: aimu.ToolContext[ResearchDeps], doi: str) -> str:  # PROPOSED (#1)
    """Save a paper you have confirmed relevant, identified by its DOI."""
    article = ctx.deps.seen.get(doi)
    if not article:
        return f"Unknown DOI {doi}. Search for it first."
    ctx.deps.store.write(f"/{doi.replace('/', '_')}.md", f"# {article['title']}\n\n{article['abstract']}")
    return f"Saved: {article['title']}"


deps = ResearchDeps(store=DocumentStore(persist_path=str(paths.output / "literature_research" / "aimu_store")))
researcher = Agent(aimu.client(MODEL), system_message=shared.RESEARCHER_SYSTEM,
                   tools=[search_preprints, save_relevant_paper, read_saved_papers], name="researcher")

aimu.pretty_print(researcher.run(shared.RESEARCH_QUESTION, deps=deps, stream=True))  # PROPOSED (#1, #4)
```

---

## What the sketch tells us

| Proposal | Friction it removes | Verdict from the sketch |
|---|---|---|
| #1 tool context | module-global `_seen` + `store`; the `shared.search_preprints(q, _seen)` hand-off | **Worth it.** Even without #3, the globals vanish and tools become unit-testable. The `ToolContext[Deps]` generic reads cleanly and keeps the model-facing schema unchanged. |
| #2 typed verdict | `pass_keyword="PASS"` and the "reply with EXACTLY PASS" prompt contortion | **Worth it, small.** `verdict_schema=Verdict` + `verdict.passed` is the natural use of AIMU's existing `schema=` support. Frees `CRITIC_SYSTEM` to be written for humans. |
| #3 corpus helper | three near-identical tools, the cache, and the all-docs `read` that doesn't scale | **Highest leverage, but most opinionated.** Collapses ~50 lines to one factory call and gives top-k retrieval for free. Risk: the factory's assumptions (what a "hit" looks like, how relevance gating works) may be too rigid; needs a `format_hit`/`gate` escape hatch. |
| #4 pretty_print | the `stream_run` chunk-interpreter | **Nice-to-have.** Pure boilerplate removal; not use-case-specific. |

**Dependency to note:** #3 leans on #1 to hide the cache inside `corpus`. Build #1 first; #3 without
it would just relocate the global into the helper.

**Borrowed-construct lineage (for the record):** the `ctx`/`deps` shape echoes **PydanticAI's
`RunContext[Deps]`** and **LangChain's `InjectedToolArg`**; `verdict_schema` echoes **PydanticAI
typed outputs**; `read_top_k` echoes **LlamaIndex's `as_query_engine(similarity_top_k=...)`**.
