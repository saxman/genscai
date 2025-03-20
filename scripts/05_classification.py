from genscai.models import HuggingFaceClient
from genscai import paths
from genscai.classification import classify_papers
from genscai.data import load_midas_data
from genscai.prompts import PromptCatalog
import pandas as pd

MODEL_KWARGS = {
    "low_cpu_mem_usage": True,
    "device_map": "balanced",
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

MODEL_ID = HuggingFaceClient.MODEL_LLAMA_3_1_8B


def run_classification():
    df_train, df_test, df_validate = load_midas_data()
    df = pd.concat([df_train, df_test, df_validate])

    model_client = HuggingFaceClient(MODEL_ID, MODEL_KWARGS)

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt = catalog.retrieve_last(MODEL_ID)

    df = classify_papers(
        model_client, prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE, CLASSIFICATION_GENERATE_KWARGS, df
    )

    df = df.query("predict_modeling == True")
    df.to_json(paths.data / "modeling_papers.json", orient="records", lines=True)

    del model_client


if __name__ == "__main__":
    run_classification()
