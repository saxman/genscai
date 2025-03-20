import logging

from genscai import paths
from genscai.models import HuggingFaceClient as ModelClient
from genscai.data import load_classification_training_data
from genscai.classification import classify_papers, test_classification
from genscai.prompts import PromptCatalog, Prompt

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ID = ModelClient.MODEL_GEMMA_3_12B

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

MUTATION_GENERATE_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.75,
    "top_k": 50,
    "top_p": 0.95,
}

TASK_PROMPT_TEMPLATE = """
Read the following scientific paper abstract. Based on the content, determine if the paper explicitly refers to or uses a disease modeling technique,
including but not limited to mathematical, statistical, or computational methods used to simulate, analyze, predict, or interpret the dynamics of a disease,
specifically in the context of estimating the probability of disease resurgence.
""".strip()

TASK_PROMPT_IO_TEMPLATE = """
If the abstract explicitly describes or references a disease modeling technique, answer "YES".
If the abstract does not explicitly describe or reference a disease modeling technique, or it focuses on non-modeling analysis, answer "NO".
Do not include any additional text or information with your response.

Abstract:
{abstract}
""".strip()

MUTATION_POS_PROMPT_TEMPLATE = """
Read the language model prompt and scientific paper abstract below. Expand the prompt so that a language model would correctly determine that the abstract
explicitly refers to or uses a disease modeling technique.

Do not include the names of specific diseases in the prompt. Do not include the abstract in the prompt.

Only return the modified prompt. Do not include any additional text or information.

Prompt:
{prompt}

Abstract:
{abstract}
""".strip()

MUTATION_NEG_PROMPT_TEMPLATE = """
Read the language model prompt and scientific paper abstract below. Expand the prompt so that a language model would correctly determine that the abstract
DOES NOT explicitly refer to or use a disease modeling technique.

Do not include the names of specific diseases in the prompt. Do not include the abstract in the prompt.

Only return the modified prompt. Do not include any additional text or information.

Prompt:
{prompt}

Abstract:
{abstract}
""".strip()


def run_training():
    """
    Executes the training process (prompt auto tuning) for classification using a hill climbing approach to optimize the prompt for 100% accuracy.

    The function logs detailed information about the process and results, including the prompt templates, classification results, and metrics.

    Raises:
        Any exceptions raised during the execution of the training process.
    """

    logging.basicConfig(filename="training.log", level=logging.INFO)
    logger.info(f"started: {MODEL_ID}")

    df_data = load_classification_training_data()
    df_data["predict_modeling"] = None

    print(f"loading model: {MODEL_ID}", flush=True)
    model_client = ModelClient(MODEL_ID, MODEL_KWARGS)

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt = catalog.retrieve_last(MODEL_ID)

    if prompt is None:
        prompt = Prompt(prompt=TASK_PROMPT_TEMPLATE, model_id=MODEL_ID, version=1)
        print("using default prompt template")
        logger.info(f"using default prompt template:\n{prompt.prompt}")
    else:
        print("using stored prompt template")
        logger.info(f"using stored prompt template:\n{prompt.prompt}")

    last_prompt = prompt
    iteration = mutation = 0

    # iterate using a basic hill climbing approach until a prompt is found that classifies w/100% accuracy
    while True:
        # don't test model if already have test metrics
        if prompt.metrics is None:
            prompt_str = prompt.prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE
            logger.info(f"task prompt template:\n{prompt_str}")

            df_data = classify_papers(model_client, prompt_str, CLASSIFICATION_GENERATE_KWARGS, df_data)
            df_data, prompt.metrics = test_classification(df_data)

            results_str = " ".join(map(str, df_data.predict_modeling))
            logger.info(f"test output:\n{results_str}")

        out = f"results: iteration: {iteration}, mutation: {mutation} - precision: {prompt.metrics['precision']:.2f}. recall: {prompt.metrics['recall']:.2f}, accuracy: {prompt.metrics['accuracy']:.2f}"
        print(out)
        logger.info(out)

        iteration += 1

        # if accuracy has decreased, revert to the last prompt and try again
        # this needs to be strictly less than, since first iteration, prompt == last_prompt
        if prompt.metrics["accuracy"] < last_prompt.metrics["accuracy"]:
            print("reverting to last prompt")
            prompt = last_prompt
            mutation -= 1
            continue

        df_bad = df_data.query("is_modeling != predict_modeling")

        # stop iterating once accuracy is 100%
        if len(df_bad) == 0:
            break

        # randomly select an incorrect result for mutating the task prompt
        item = df_bad.sample().iloc[0]
        if item.is_modeling:
            mutation_prompt = MUTATION_POS_PROMPT_TEMPLATE.format(prompt=prompt.prompt, abstract=item.abstract)
        else:
            mutation_prompt = MUTATION_NEG_PROMPT_TEMPLATE.format(prompt=prompt.prompt, abstract=item.abstract)

        print("mutating prompt")
        logger.info(f"mutation prompt:\n{mutation_prompt}")

        # generate a new prompt using the mutation prompt
        mutated_prompt = model_client.generate_text(mutation_prompt, MUTATION_GENERATE_KWARGS).strip()

        logger.info(f"mutated task prompt:\n{mutated_prompt}")

        mutation += 1

        # store the improved prompt, then re-assign
        catalog.store_prompt(prompt)
        last_prompt = prompt

        prompt = Prompt(model_id=MODEL_ID, version=last_prompt.version + 1, prompt=mutated_prompt)

    logger.debug(f"finished: {MODEL_ID}")


if __name__ == "__main__":
    run_training()
