"""Index downloaded medRxiv articles into the Chroma knowledge base (output/medrxiv.db).

Thin CLI over genscai.knowledge_base.build_knowledge_base. Run after 01_medrxiv_download.py. The
resulting vector store backs the MCP search tool and the RAG / agents / evaluation use cases.
"""

from genscai import knowledge_base

if __name__ == "__main__":
    years = list(range(2019, 2025))
    print("Indexing medRxiv articles into the Chroma knowledge base...")
    count = knowledge_base.build_knowledge_base(years, chunked=True)
    print(f"Knowledge base now holds {count} chunks in collection '{knowledge_base.CHUNKED_COLLECTION}'")
