import pandas as pd
import os.path

from genscai.models import HuggingFaceClient
from genscai import paths
from genscai.models import classify_papers
from genscai.prompts import PromptCatalog

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

def run_classification():
    fname = paths.data / "train.json"
    if os.path.isfile(fname):
        df_train = pd.read_json(fname)
    else:
        df_train = pd.read_csv("hf://datasets/krosenf/midas-abstracts/train.csv")
        df_train.to_json(fname)

    fname = paths.data / "validate.json"
    if os.path.isfile(fname):
        df_validate = pd.read_json(fname)
    else:
        df_validate = pd.read_csv("hf://datasets/krosenf/midas-abstracts/validate.csv")
        df_validate.to_json(fname)

    fname = paths.data / "test.json"
    if os.path.isfile(fname):
        df_test = pd.read_json(fname)
    else:
        df_test = pd.read_csv("hf://datasets/krosenf/midas-abstracts/test.csv")
        df_test.to_json(fname)

    df = df_validate
    df = pd.concat([df_train, df_validate, df_test])

    model_client = HuggingFaceClient(HuggingFaceClient.MODEL_LLAMA_3_1_8B, MODEL_KWARGS)

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt = catalog.retrieve_last(HuggingFaceClient.MODEL_LLAMA_3_1_8B)

    df = classify_papers(
        model_client, prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE, CLASSIFICATION_GENERATE_KWARGS, df
    )

    del model_client

if __name__ == "__main__":
    run_classification()