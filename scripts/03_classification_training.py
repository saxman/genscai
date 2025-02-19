import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)

from genscai import paths
from genscai.models import OllamaClient as ModelClient
from genscai.prompts import PromptCatalog, Prompt

MODEL_ID = ModelClient.MODEL_LLAMA_3_2_3B

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

MUTATION_GENERATE_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.75,
    "top_k": 50,
    "top_p": 0.95,
}

TASK_PROMPT_TEMPLATE = """
Read the following scientific paper abstract. Based on the content, determine if the paper explicitly refers to or uses a disease modeling technique, including but not limited to mathematical, statistical, or computational methods used to simulate, analyze, predict, or interpret the dynamics of a disease, specifically in the context of estimating the probability of disease resurgence.
"""

TASK_PROMPT_IO_TEMPLATE = """
If the abstract explicitly describes or references a disease modeling technique, answer "YES".
If the abstract does not explicitly describe or reference a disease modeling technique, or it focuses on non-modeling analysis, answer "NO".
Do not include any additional text or information with your response.

Abstract:
{abstract}
"""

MUTATION_POS_PROMPT_TEMPLATE = """
Read the language model prompt and scientific paper abstract below. Expand the prompt so that a language model would correctly determine that the abstract explicitly refers to or uses a disease modeling technique.

Do not include the names of specific diseases in the prompt.

Only return the modified prompt. Do not include any additional text or information.

Prompt:
{prompt}

Abstract:
{abstract}
"""

MUTATION_NEG_PROMPT_TEMPLATE = """
Read the language model prompt and scientific paper abstract below. Expand the prompt so that a language model would correctly determine that the abstract DOES NOT explicitly refer to or use a disease modeling technique.

Do not include the names of specific diseases in the prompt.

Only return the modified prompt. Do not include any additional text or information.

Prompt:
{prompt}

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
) -> pd.DataFrame:
    results = {}
    predict_modeling = []

    for paper in df_data.itertuples():
        print(".", end="", flush=True)

        prompt = prompt_template.format(abstract=paper.abstract)
        result = model_client.generate_text(prompt, CLASSIFICATION_GENERATE_KWARGS)

        if "yes" in result.lower():
            predict_modeling.append(True)
        elif "no" in result.lower():
            predict_modeling.append(False)
        else:
            predict_modeling.append(pd.NA)

    print()

    df_data["predict_modeling"] = predict_modeling

    return df_data


def run_training():
    logging.basicConfig(filename='training.log', level=logging.INFO)
    logger.info('started')

    df_data = load_test_data()

    print('loading model...', flush=True)
    model_client = ModelClient(MODEL_ID, MODEL_KWARGS)

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt = catalog.retrieve_last(MODEL_ID)
    if prompt is None:
        prompt = Prompt(prompt=TASK_PROMPT_TEMPLATE, model_id=MODEL_ID, version=1)
        logger.info('using default prompt')
    else:
        logger.info('using stored prompt')
    logger.info(prompt.prompt)

    last_prompt = prompt
    i = m = 0

    # iterate using a basic hill climbing approach until a prompt is found that classifies w/100% accuracy
    while True:
        df_data = test_classification(
            model_client, prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE, df_data
        )

        true_pos = len(
            df_data.query("is_modeling == True and predict_modeling == True")
        )
        true_neg = len(
            df_data.query("is_modeling == False and predict_modeling == False")
        )

        accuracy = (true_pos + true_neg) / len(df_data)
        pos = len(df_data.query("predict_modeling == True"))
        precision = true_pos / pos if pos > 0 else 0
        pos = len(df_data.query("is_modeling == True"))
        recall = true_pos / pos if pos > 0 else 0

        prompt.metrics = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }

        out = f"iteration: {i}, mutation: {m} - precision: {precision:.2f}. recall: {recall:.2f}, accuracy: {accuracy:.2f}"
        print(out)
        logger.info(out)

        i += 1

        # if accuracy has decreased, revert to the last prompt and try again
        # this needs to be strictly less than, since first iteration, prompt == last_prompt
        if prompt.metrics["accuracy"] < last_prompt.metrics["accuracy"]:
            prompt = last_prompt
            m -= 1
            continue

        # the prompt is improved: store it, then mutate it
        catalog.store_prompt(prompt)
        last_prompt = prompt

        df_bad = df_data.query("is_modeling != predict_modeling")

        # stop iterating once accuracy is 100%
        if len(df_bad) == 0:
            break

        m += 1

        # randomly select an incorrect result for mutating the task prompt
        item = df_bad.sample().iloc[0]
        if item.is_modeling:
            mutation_prompt = MUTATION_POS_PROMPT_TEMPLATE.format(
                prompt=last_prompt.prompt, abstract=item.abstract
            )
        else:
            mutation_prompt = MUTATION_NEG_PROMPT_TEMPLATE.format(
                prompt=last_prompt.prompt, abstract=item.abstract
            )

        logger.info("mutating prompt")
        logger.info(mutation_prompt)

        # generate a new prompt using the mutation
        mutated_prompt = model_client.generate_text(mutation_prompt, MUTATION_GENERATE_KWARGS).strip()
        prompt = Prompt(
            model_id=MODEL_ID,
            version=last_prompt.version + 1,
            prompt=mutated_prompt
        )

        logger.info("mutated prompt")
        logger.info(mutated_prompt)

logger.debug('finished')

if __name__ == "__main__":
    run_training()
