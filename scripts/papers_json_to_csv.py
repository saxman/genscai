from genscai import paths
from genscai.data import load_midas_data
import pandas as pd
from pathlib import Path

# Define file paths
modeling_papers_path = paths.data / "modeling_papers_0.json"

# Output CSV file paths
modeling_papers_csv = paths.output / "modeling_papers.csv"
non_modeling_papers_csv = paths.output / "non_modeling_papers.csv"


def main():
    """
    Create CSV files for modeling and non-modeling papers.
    """

    df_modeling_papers = pd.read_json(modeling_papers_path, orient="records", lines=True)

    # Load all MIDAS papers
    df_train, df_test, df_validate = load_midas_data()
    df_all_papers = pd.concat([df_train, df_test, df_validate])

    # Create dataframe for non-modeling papers by excluding modeling papers from all papers
    df_non_modeling_papers = df_all_papers[~df_all_papers["id"].isin(df_modeling_papers["id"])]

    # Ensure the output directory exists
    Path(paths.output).mkdir(parents=True, exist_ok=True)

    df_modeling_papers.to_csv(modeling_papers_csv, index=False)
    df_non_modeling_papers.to_csv(non_modeling_papers_csv, index=False)


if __name__ == "__main__":
    main()
