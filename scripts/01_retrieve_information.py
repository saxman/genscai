import genscai

# Create a retriever object
for year in range(2020, 2024):
    print(f"Retrieving articles for {year}")
    retriever = genscai.retrieval.MIDASRetriever(startdate=f'{year}-01-01', enddate=f'{year}-12-31')
    retriever()