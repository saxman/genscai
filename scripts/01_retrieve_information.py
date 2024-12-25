"""
This script retrieves MIDAS abstracts and metadata using a daterange (inclusive) 
and saves them to a database.
"""
import genscai

OUTPUT_PATH = genscai.paths.data / 'raw'

# Create and run a retriever ($$)
for year in range(2020, 2024):
    print(f"Retrieving articles for {year}")
    retriever = genscai.retrieval.MIDASRetriever(startdate=f'{year}-01-01', enddate=f'{year}-12-31', database_path=OUTPUT_PATH)
    retriever()