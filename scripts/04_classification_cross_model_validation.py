import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

from genscai import paths
from genscai.models import HuggingFaceClient as ModelClient
from genscai.prompts import PromptCatalog, Prompt

MODEL_KWARGS = {
    "low_cpu_mem_usage": True,
    "device_map": "sequential",  # load the model into GPUs sequentially, to avoid memory allocation issues with balancing
    "torch_dtype": "auto",
}

CLASSIFICATION_GENERATE_KWARGS = {
    "max_new_tokens": 1,
    "temperature": 0.01,
    "do_sample": True,
}

TASK_PROMPT_IO_TEMPLATE = """
If the abstract explicitly describes or references a disease modeling technique, answer "YES".
If the abstract does not explicitly describe or reference a disease modeling technique, or it focuses on non-modeling analysis, answer "NO".
Do not include any additional text or information with your response.

Abstract:
{abstract}
"""


def load_test_data() -> pd.DataFrame:
    with open(paths.data / "modeling_papers.json", "r") as f:
        data = json.load(f)

    df1 = pd.json_normalize(data)
    df1["is_modeling"] = True

    with open(paths.data / "non_modeling_papers.json", "r") as f:
        data = json.load(f)

    df2 = pd.json_normalize(data)
    df2["is_modeling"] = False

    return pd.concat([df1, df2])


def test_classification(
    model_client: ModelClient, prompt_template: str, df_data: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    predict_modeling = []

    print("testing prompt: ", end="")

    for paper in df_data.itertuples():
        prompt = prompt_template.format(abstract=paper.abstract)
        result = model_client.generate_text(prompt, CLASSIFICATION_GENERATE_KWARGS)

        if "yes" in result.lower():
            predict_modeling.append(True)
        elif "no" in result.lower():
            predict_modeling.append(False)
        else:
            predict_modeling.append(pd.NA)

        print(".", end="", flush=True)

    print()

    df_data["predict_modeling"] = predict_modeling

    true_pos = len(df_data.query("is_modeling == True and predict_modeling == True"))
    true_neg = len(df_data.query("is_modeling == False and predict_modeling == False"))

    accuracy = (true_pos + true_neg) / len(df_data)
    pos = len(df_data.query("predict_modeling == True"))
    precision = true_pos / pos if pos > 0 else 0
    pos = len(df_data.query("is_modeling == True"))
    recall = true_pos / pos if pos > 0 else 0

    return df_data, {"accuracy": accuracy, "precision": precision, "recall": recall}


def run_training():
    logging.basicConfig(filename="validation.log", level=logging.INFO)

    df_data = load_test_data()
    df_data["predict_modeling"] = None

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt_model_ids = catalog.retrieve_model_ids()
    test_model_ids = [x for x in prompt_model_ids if '/' in x]

    for test_model_id in test_model_ids:
        print(f"loading model: {test_model_id}", flush=True)
        model_client = ModelClient(test_model_id, MODEL_KWARGS)

        for prompt_model_id in prompt_model_ids:
            prompt = catalog.retrieve_last(prompt_model_id)

            df_data, metrics = test_classification(
                model_client, prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE, df_data
            )

            print(f'prompt: {prompt_model_id}, model: {test_model_id}, metrics: {metrics}')
        
        del model_client


if __name__ == "__main__":
    run_training()
