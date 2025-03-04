import logging

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

from genscai import paths
from genscai.models import HuggingFaceClient as ModelClient
from genscai.data import load_classification_training_data
from genscai.classification import classify_papers, test_classification
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


def run_tests():
    logging.basicConfig(filename="validation.log", level=logging.INFO)

    df_data = load_classification_training_data()
    df_data["predict_modeling"] = None

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt_model_ids = catalog.retrieve_model_ids()
    test_model_ids = [x for x in prompt_model_ids if "/" in x]

    for test_model_id in test_model_ids:
        print(f"loading model: {test_model_id}", flush=True)
        model_client = ModelClient(test_model_id, MODEL_KWARGS)

        for prompt_model_id in prompt_model_ids:
            prompt = catalog.retrieve_last(prompt_model_id)
            prompt_template = prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE

            df_data = classify_papers(model_client, prompt_template, CLASSIFICATION_GENERATE_KWARGS, df_data)
            metrics = test_classification(df_data)

            print(f"prompt: {prompt_model_id}, model: {test_model_id}, metrics: {metrics}")

        del model_client


if __name__ == "__main__":
    run_tests()
