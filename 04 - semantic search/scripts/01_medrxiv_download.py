"""Download medRxiv articles (2019–2024) to output/medrxiv_{year}.json.

Thin CLI over genscai.knowledge_base.download_articles. Run before 02_medrxiv_knowledge_base.py.
"""

from genscai import knowledge_base

if __name__ == "__main__":
    years = list(range(2019, 2025))
    print(f"Downloading medRxiv articles for {years[0]}–{years[-1]}...")
    total = knowledge_base.download_articles(years)
    print(f"Wrote {total} articles across {len(years)} years to output/")
