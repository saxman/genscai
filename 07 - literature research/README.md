# Literature Research

The **agentic literature research** use case, implemented **six times** in [`notebooks/`](notebooks), once per agent framework, all running on **local models**, by default, via [Ollama](https://ollama.com). Because only the framework changes, you can read the notebooks side by side to see how each expresses the same agentic workflow.

Unlike the numbered tutorial folders, these notebooks are siblings to be **compared**, not a sequence to work through in order.

## The use case

Given a research question, each notebook's agent:

1. **Searches live sources** — medRxiv and bioRxiv preprints, queried by keyword through the
   [Europe PMC](https://europepmc.org/) REST API (it indexes both servers and returns abstracts
   reliably, with no scraping or pre-built database).
2. **Confirms relevance, then stores** — the agent judges each hit against the question and writes
   **only the matches** into a **local document store**. The store is the justified local artifact: a
   curated, reusable corpus of vetted papers, gated so off-topic hits never land.
3. **Synthesizes** a cited answer from the curated corpus.
4. **Evaluates and loops** — a critic reviews the synthesis and either accepts it or feeds specific
   gaps back for another round (bounded by a max-rounds cap).

The shared search/fetch functions live in [`genscai/research.py`](../genscai/research.py) and are
identical across all six notebooks. Everything else — agent construction, tool binding, the document
store, and how the feedback loop is expressed — is the framework's own idiom.

## The notebooks

| # | Framework | Document store | Feedback loop idiom |
|---|---|---|---|
| [1](notebooks/01%20-%20AIMU.ipynb) | **AIMU** | `aimu.memory.DocumentStore` (native, file-backed) | `EvaluatorOptimizer` workflow |
| [2](notebooks/02%20-%20smolagents.ipynb) | **smolagents** | ChromaDB collection via tools (no native store) | hand-written Python loop |
| [3](notebooks/03%20-%20LangGraph.ipynb) | **LangGraph** | LangChain `Chroma` vector store | `StateGraph` conditional edge |
| [4](notebooks/04%20-%20PydanticAI.ipynb) | **PydanticAI** | ChromaDB collection via tools (no native store) | hand-written loop + critic agent |
| [5](notebooks/05%20-%20CrewAI.ipynb) | **CrewAI** | ChromaDB collection via tools (Knowledge noted) | multi-agent crew + loop |
| [6](notebooks/06%20-%20LlamaIndex.ipynb) | **LlamaIndex** | `VectorStoreIndex` (native) | Workflow events |

The **document store** column is a deliberate axis of comparison: frameworks with a native store use
it (AIMU, LangChain/LangGraph, LlamaIndex); those without one (smolagents, PydanticAI) use an external
ChromaDB collection, the idiomatic choice for each. CrewAI has a native Knowledge feature, but it is
fixed at agent-creation time and does not fit gated mid-run curation, so it uses tools too (the
notebook explains and sketches the Knowledge alternative).

## Prerequisites

Install the agent frameworks (in addition to the base project):

```bash
uv sync --all-extras   # includes the `agents` extra
```

Pull a tool-capable chat model and an embedding model with Ollama:

```bash
ollama pull qwen3.6:27b      # default; reliable at multi-step tool use
ollama pull nomic-embed-text # embeddings for the vector-backed stores
```

## Choosing a model

The notebooks default to `qwen3.6:27b`. Multi-step, open-ended research is demanding: a 27B-class
model curates papers and writes grounded syntheses reliably, while smaller models (e.g. `qwen3.5:9b`)
run faster but more often drift off-topic or under-cite.

Override the model without editing the notebooks via an environment variable (all six honor it):

```bash
GENSCAI_AGENT_MODEL=qwen3.5:9b jupyter lab
```

## A note on local-inference reliability

These are demonstrations of *framework mechanics*, not benchmarks. Local models are
non-deterministic and vary run to run — the agent may save a different set of papers or word its
synthesis differently each time. The relevance gate is the model's own judgment, so corpus contents
will vary. Larger models reduce this variance.
