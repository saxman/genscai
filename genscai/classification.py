import pandas as pd
from tqdm import tqdm

from genscai.models import ModelClient


def classify_papers(
    model_client: ModelClient,
    prompt_template: str,
    generate_kwargs,
    df_data: pd.DataFrame,
) -> pd.DataFrame:
    predict_modeling = []

    for i in tqdm(range(len(df_data)), desc="classifying"):
        paper = df_data.iloc[i]
        prompt = prompt_template.format(abstract=paper.abstract)
        result = model_client.generate_text(prompt, generate_kwargs)

        if "yes" in result.lower():
            predict_modeling.append(True)
        elif "no" in result.lower():
            predict_modeling.append(False)
        else:
            predict_modeling.append(pd.NA)

    df_data["predict_modeling"] = predict_modeling

    return df_data


def test_classification(
    df_data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    true_pos = len(df_data.query("is_modeling == True and predict_modeling == True"))
    true_neg = len(df_data.query("is_modeling == False and predict_modeling == False"))

    accuracy = (true_pos + true_neg) / len(df_data)
    pos = len(df_data.query("predict_modeling == True"))
    precision = true_pos / pos if pos > 0 else 0
    pos = len(df_data.query("is_modeling == True"))
    recall = true_pos / pos if pos > 0 else 0

    return df_data, {"accuracy": accuracy, "precision": precision, "recall": recall}
