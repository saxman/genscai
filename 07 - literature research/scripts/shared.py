"""Shared constants, prompts, and helpers for the literature-research scripts (01-06).

Every script in this folder implements the same agentic literature-research workflow on a different
framework. The framework-specific wiring is what's worth comparing; everything in this module is the
boilerplate they have in common, kept here so each script reads as just its framework's code.

Importing this module loads the project `.env` so the scripts don't each repeat that line.
"""

import os

import dotenv

from genscai import research, paths

dotenv.load_dotenv(paths.root / ".env")

DEFAULT_MODEL = "qwen3.6:27b"
OLLAMA_URL = "http://localhost:11434"
MAX_ROUNDS = 2

RESEARCH_QUESTION = (
    "What interventions and control strategies have recent preprints proposed or evaluated for "
    "dengue outbreaks, and what evidence do they report?"
)

# Workflow prompt shared by the frameworks that drive a single tool-calling agent (smolagents,
# LangGraph, PydanticAI). AIMU, CrewAI, and LlamaIndex need framework-shaped variants and keep
# their own inline.
RESEARCHER_SYSTEM = (
    "You are a research assistant building a cited literature review grounded in a curated corpus. "
    "Workflow: (1) call read_saved_papers to see what is curated; (2) use search_preprints, always "
    "including the key topic term (e.g. 'dengue') in queries; (3) for each candidate, if its abstract "
    "is plausibly relevant call save_relevant_paper(doi), erring toward saving; (4) before synthesizing "
    "you MUST call read_saved_papers and base the synthesis only on those papers, citing each by title "
    "and DOI. Never claim no papers were found if the store is non-empty. Be concise."
)

CRITIC_SYSTEM = (
    "You review a literature synthesis. Reply with exactly PASS if it answers the question, cites "
    "specific papers (title + DOI), and reports evidence. Otherwise reply REVISE: <specific gap>."
)

# For the AIMU scripts, which use a typed Verdict(passed, feedback) instead of matching "PASS".
CRITIC_VERDICT_SYSTEM = (
    "You review a literature synthesis built from a curated corpus of saved preprints. Set passed=true "
    "when it answers the question, cites specific papers (title + DOI), and reports evidence from them; "
    "otherwise set passed=false and put the single most important missing piece in feedback. Do not "
    "demand topics outside the saved corpus, and do not penalize a focused answer."
)


def agent_model():
    """Return the Ollama model name, overridable via GENSCAI_AGENT_MODEL. Each script adds the
    provider prefix its framework expects (e.g. 'ollama:', 'ollama_chat/', 'ollama/')."""
    return os.environ.get("GENSCAI_AGENT_MODEL", DEFAULT_MODEL)


def print_search_preview(query="dengue vaccination", max_results=3, show_doi=False):
    """Print a few medRxiv hits so a run shows the search backend is reachable before the agent starts."""
    for hit in research.search_medrxiv(query, max_results=max_results):
        print(hit["date"], "-", hit["title"][:80])
        if show_doi:
            print("   ", hit["doi"])


def search_preprints(query, seen):
    """Search medRxiv + bioRxiv, cache each hit in `seen` keyed by DOI (so a paper can later be saved
    by DOI alone), and return one formatted block per hit. This is the body of every script's
    search_preprints tool; the scripts wrap it in their framework's tool decorator."""
    try:
        results = research.search_medrxiv(query, max_results=5) + research.search_biorxiv(query, max_results=3)
    except Exception as exc:
        return f"Search temporarily unavailable ({exc}). Try again shortly or rephrase the query."
    if not results:
        return "No results found."
    blocks = []
    for article in results:
        seen[article["doi"]] = article
        blocks.append(
            f"DOI: {article['doi']}\nTitle: {article['title']}\nDate: {article['date']}\n"
            f"Abstract: {(article['abstract'] or '')[:600]}"
        )
    return "\n\n".join(blocks)


def format_saved_paper(title, doi, date, url, body):
    """Render one saved paper for read_saved_papers, shared by the vector-store-backed scripts."""
    return f"# {title}\nDOI: {doi} | Date: {date}\nURL: {url}\n\n{body}"


def print_saved_summary(titles):
    """Print the final 'N papers saved' list shared by the scripts that end with a store dump."""
    print(f"{len(titles)} papers saved:")
    for title in titles:
        print(" -", title[:80])
