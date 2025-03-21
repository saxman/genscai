import logging

from genscai import paths
from genscai.models import MODEL_KWARGS, HuggingFaceClient as ModelClient
from genscai.data import load_classification_training_data
from genscai.prompts import PromptCatalog
import genscai.classification as gc

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

CLASSIFICATION_GENERATE_KWARGS = gc.CLASSIFICATION_GENERATE_KWARGS.copy()
CLASSIFICATION_GENERATE_KWARGS.update(
    {
        "max_new_tokens": 1024,
        "temperature": 0.70,
    }
)


def run_tests():
    """
    Validates how well a prompt tuned by a particular model performs on classification tasks using different models.
    """

    logging.basicConfig(filename="validation.log", level=logging.INFO)

    df_data = load_classification_training_data()
    df_data["predict_modeling"] = None

    catalog = PromptCatalog(paths.data / "prompt_catalog.db")
    prompt_model_ids = catalog.retrieve_model_ids()

    # only test models that are local Hugging Face models, which are identified by having a "/" in the model ID
    test_model_ids = [x for x in prompt_model_ids if "/" in x]

    # TODO only running for one model for now
    test_model_ids = [ModelClient.MODEL_DEEPSEEK_R1_8B]

    for test_model_id in test_model_ids:
        print(f"loading model: {test_model_id}", flush=True)
        model_client = ModelClient(test_model_id, MODEL_KWARGS)

        for prompt_model_id in prompt_model_ids:
            prompt = catalog.retrieve_last(prompt_model_id)
            prompt_template = prompt.prompt + gc.CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE

            df_data = gc.classify_papers(model_client, prompt_template, CLASSIFICATION_GENERATE_KWARGS, df_data)
            df_data, metrics = gc.test_paper_classifications(df_data)

            print(f"prompt: {prompt_model_id}, model: {test_model_id}, metrics: {metrics}")

        del model_client


if __name__ == "__main__":
    run_tests()
