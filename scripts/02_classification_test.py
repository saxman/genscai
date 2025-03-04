import pandas as pd

from genscai import paths
from genscai.models import OllamaClient as ModelClient
from genscai.data import load_classification_training_data
from genscai.classification import test_classification, classify_papers

MODEL_ID = ModelClient.MODEL_LLAMA_3_1_8B

MODEL_KWARGS = {
    "low_cpu_mem_usage": True,
    "device_map": "sequential",  # load the model into GPUs sequentially, to avoid memory allocation issues with balancing
    "torch_dtype": "auto",
}

CLASSIFICATION_GENERATE_KWARGS = {"max_new_tokens": 1, "temperature": 0.01, "do_sample": True}

TASK_PROMPT_TEMPLATE = """
Read the following scientific paper abstract. Based on the content, determine if the paper explicitly refers to or uses a disease modeling technique, including but not limited to mathematical, statistical, or computational methods used to simulate, analyze, predict, or interpret the dynamics of a disease, specifically in the context of estimating the probability of disease resurgence.

Consider the use of disease modeling if the abstract describes or references compartmental models, statistical models, simulation models, mathematical equations, or functional forms to analyze or predict disease transmission, risk factors, or the effects of interventions.

Additionally, if the paper uses epidemiological modeling, disease forecasting, regression analysis, or statistical analysis to investigate associations between disease characteristics and external factors, consider it a form of disease modeling technique.

If the abstract specifically mentions estimating the probability of disease resurgence using quantitative methods, such as statistical models or mathematical equations, consider it a form of disease modeling technique.

If the abstract discusses statistical models or techniques for analyzing topics unrelated to disease dynamics or epidemiology, such as software defects or unrelated fields, do not consider it a form of disease modeling technique.
"""

TASK_PROMPT_IO_TEMPLATE = """
If the abstract describes or references any of these methods or similar approaches, answer "YES".
If the abstract focuses on non-modeling analysis, such as reporting observational data without reference to disease modeling techniques, answer "NO".
Do not include any additional text or information.

Abstract:
{abstract}
"""


def run_test():
    df_data = load_classification_training_data()
    model_client = ModelClient(MODEL_ID, MODEL_KWARGS)
    prompt_template = TASK_PROMPT_TEMPLATE + "\n\n" + TASK_PROMPT_IO_TEMPLATE

    df_data = classify_papers(model_client, prompt_template, CLASSIFICATION_GENERATE_KWARGS, df_data)
    metrics = test_classification(df_data)

    print(
        f"results: precision: {metrics['precision']:.2f}. recall: {metrics['recall']:.2f}, accuracy: {metrics['accuracy']:.2f}"
    )
    print(df_data.query("is_modeling != predict_modeling"))

    del model_client


if __name__ == "__main__":
    run_test()
