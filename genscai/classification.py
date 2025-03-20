import pandas as pd
from tqdm import tqdm
import logging

from genscai.models import ModelClient

logger = logging.getLogger(__name__)

CLASSIFICATION_GENERATE_KWARGS = {
    "max_new_tokens": 1,
    "temperature": 0.01,
    "do_sample": True
}

CLASSIFICATION_TASK_PROMPT_TEMPLATE = """
Read the following scientific paper abstract. Based on the content, determine if the paper explicitly refers to or uses a disease modeling technique,
including but not limited to mathematical, statistical, or computational methods used to simulate, analyze, predict, or interpret the dynamics of a disease.
"""

CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE = """
If the abstract describes or references any of these methods or similar approaches, answer "YES".
If the abstract focuses on non-modeling analysis, such as reporting observational data without reference to disease modeling techniques, answer "NO".
Wrap your final answer in square braces, e.g. [YES] or [NO].

Abstract:
{abstract}
"""


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

        logger.info(f"classification result: {result}")

        if "[yes]" in result.lower():
            predict_modeling.append(True)
        elif "[no]" in result.lower():
            predict_modeling.append(False)
        else:
            predict_modeling.append(pd.NA)
        
        logger.info(f"classification: {predict_modeling[-1]}")

    df_data["predict_modeling"] = predict_modeling

    return df_data


def test_paper_classifications(
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
