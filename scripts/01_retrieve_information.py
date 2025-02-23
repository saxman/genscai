"""
This script retrieves MIDAS abstracts and metadata using a daterange (inclusive)
and saves them to a database.
"""

import hashlib
import sciris as sc
import polars as pl
from icecream import ic
from pathlib import Path
from collections import defaultdict
import genscai

OUTPUT_PATH = genscai.paths.data


def run_retriever():
    # Create and run a retriever ($$)
    for year in range(2020, 2024):
        print(f"Retrieving articles for {year}")
        retriever = genscai.retrieval.MIDASRetriever(
            startdate=f"{year}-01-01",
            enddate=f"{year}-12-31",
            database_path=OUTPUT_PATH / "raw",
        )
        retriever()


def train_test_validate_split(df, test_size=0.15, validate_size=0.15, verbose=False):
    # shuffle the dataframe
    df = df.sample(fraction=1, shuffle=True)

    # determine the sizes of the train, test and validate sets
    n = len(df)
    test_size = int(0.15 * n)
    validate_size = int(0.15 * n)
    train_size = n - test_size - validate_size

    # split the dataframe into train, test and validate sets
    if test_size > 0:
        test, df = df.head(test_size), df.tail(-test_size)
    else:
        test = pl.DataFrame()
    if validate_size > 0:
        validate, df = df.head(validate_size), df.tail(-validate_size)
    else:
        validate = pl.DataFrame()

    assert len(df) == train_size, "Somthing went wrong with the train-test-validate split"
    assert len(df) + len(test) + len(validate) == n, "Somthing went wrong with the train-test-validate split"

    if verbose:
        print(f"Train set size: {len(df)}")
        print(f"Test set size: {len(test)}")
        print(f"Validate set size: {len(validate)}")

    return df, test, validate


def run_processor():
    # Process databasefiles into a single dataframe
    files = sc.getfilelist(OUTPUT_PATH / "raw", "*.json")
    print("Processing 10 files")

    # construct the dataframe from the table
    df = defaultdict(list)
    for file in files:
        dates = [sc.date(date) for date in genscai.utils.extract_dates(file)]
        years = [date.year for date in dates]
        assert len(set(years)) == 1, "All dates should be from the same year"

        db = genscai.utils.ReadOnlyTinyDB(file)
        table = db.table("articles")
        for record in table:
            for k, v in record.items():
                df[k].append(v)
            df["year"].append(years[0])
    df = pl.DataFrame(df)

    # Remove entries with no abstract or title
    df = df.filter(pl.col("abstract") != "No abstract available.")
    df = df.filter(pl.col("title") != "No abstract available.")

    # create an id
    def hash_str(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()

    # Add the hashed 'id' column
    df = df.with_columns(pl.col("title").map_elements(hash_str).alias("id"))

    # Create first author column
    df = df.with_columns(pl.col("reference").map_elements(lambda x: x.split(" ")[0]).alias("first_author"))
    # drop dublicates using first author
    df = df.unique(subset=["title"], keep="last")  # can also subset by first_author

    # identify elements of df whose ids are not unique
    duplicate_ids = df.group_by("id").count().filter(pl.col("count") > 1)
    duplicates = df.filter(pl.col("id").is_in(duplicate_ids["id"]))
    for id, group in duplicates.group_by("id"):
        print(group)

    # assert that all the ids are now unique
    print(df["id"].n_unique(), len(df))
    assert df["id"].n_unique() == len(df), "IDs are not unique"

    # reorder the columns so that id is first
    df = df.select(["id"] + [col for col in df.columns if col != "id"])

    # drop the first_author column
    df.drop_in_place("first_author")

    # split the dataframe into train, test and validate sets
    train, test, validate = train_test_validate_split(df, verbose=True)

    # save the dataframes to csv files
    output_dir = OUTPUT_PATH / "processed"
    sc.makefilepath(folder=output_dir, makedirs=True)
    train.write_csv(output_dir / "train.csv")
    test.write_csv(output_dir / "test.csv")
    validate.write_csv(output_dir / "validate.csv")


if __name__ == "__main__":
    # run_retriever() # $$
    run_processor()
