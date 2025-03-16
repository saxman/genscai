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

MODEL_IDS = [
    HuggingFaceClient.MODEL_LLAMA_3_1_8B,
    HuggingFaceClient.MODEL_GEMMA_2_9B,
    HuggingFaceClient.MODEL_PHI_4_14B,
    HuggingFaceClient.MODEL_MISTRAL_NEMO_12B,
]


def run_classification():
    df_train, df_test, df_validate = load_midas_data()
    catalog = PromptCatalog(paths.data / "prompt_catalog.db")

    for i, model_id in enumerate(MODEL_IDS):
        df = pd.concat([df_train, df_test, df_validate])

        model_client = HuggingFaceClient(model_id, MODEL_KWARGS)
        prompt = catalog.retrieve_last(model_id)

        df = classify_papers(
            model_client, prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE, CLASSIFICATION_GENERATE_KWARGS, df
        )

        df = df.query("predict_modeling == True")
        df.to_json(paths.data / f"modeling_papers_{i}.json", orient="records", lines=True)

        del model_client


if __name__ == "__main__":
    run_classification()
