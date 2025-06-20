import logging

from genscai import paths
from aimu.models import HuggingFaceClient as ModelClient
from genscai.classification import (
    CLASSIFICATION_TASK_PROMPT_TEMPLATE,
    CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE,
    classify_papers,
    test_paper_classifications,
)
from genscai.data import load_classification_training_data
from genscai import PromptCatalog, Prompt
from genscai.training import MUTATION_POS_PROMPT_TEMPLATE, MUTATION_NEG_PROMPT_TEMPLATE, MUTATION_GENERATE_KWARGS
import genscai.classification as gc

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_ID = ModelClient.MODEL_DEEPSEEK_R1_8B


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
    model_client = ModelClient(MODEL_ID)

    generate_kwargs = gc.CLASSIFICATION_GENERATE_KWARGS.copy()
    # for reasoning models (e.g. DeepSeek R1), increase temperature and max_new_tokens
    if model_client.model_id == ModelClient.MODEL_DEEPSEEK_R1_8B:
        generate_kwargs.update(
            {
                "max_new_tokens": 1024,
                "temperature": 0.70,
            }
        )

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt = catalog.retrieve_last(MODEL_ID)

    if prompt is None:
        prompt = Prompt(prompt=CLASSIFICATION_TASK_PROMPT_TEMPLATE, model_id=MODEL_ID, version=1)
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
            prompt_str = prompt.prompt + CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE
            logger.info(f"task prompt template:\n{prompt_str}")

            df_data = classify_papers(model_client, prompt_str, generate_kwargs, df_data)
            df_data, prompt.metrics = test_paper_classifications(df_data)

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
        result = model_client.generate(mutation_prompt, MUTATION_GENERATE_KWARGS)
        logger.info(f"raw prompt mutation results:\n{result}")

        # extract the prompt from the response
        mutated_prompt = result.split("<prompt>")[-1].split("</prompt>")[0].strip()
        logger.info(f"processed prompt mutation results:\n{mutated_prompt}")

        mutation += 1

        # store the improved prompt, then re-assign
        catalog.store_prompt(prompt)
        last_prompt = prompt

        prompt = Prompt(model_id=MODEL_ID, version=last_prompt.version + 1, prompt=mutated_prompt)

    logger.debug(f"finished: {MODEL_ID}")


if __name__ == "__main__":
    run_training()
