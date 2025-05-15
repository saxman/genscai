from genscai import medxriv, paths
import json


if __name__ == "__main__":
    for year in range(2019, 2025):
        start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    print(f"Retrieving articles from {start_date} to {end_date}")
    articles = medxriv.retrieve_articles(start_date=start_date, end_date=end_date)

    print(f"Writing {len(articles)} articles to file")
    with open(paths.output / f"medxriv_{year}.json", "w") as fout:
        json.dump(articles, fout)
