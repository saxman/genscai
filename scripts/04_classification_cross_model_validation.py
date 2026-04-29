import logging

from genscai import paths
from aimu.models import HuggingFaceClient
from aimu.models.hf.hf_client import HuggingFaceModel
from aimu.prompts import PromptCatalog
from genscai.data import load_classification_training_data
import genscai.classification as gc

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_NAME = "classification"


def _hf_model_for(model_id: str) -> HuggingFaceModel | None:
    """Map a stored Hugging Face model_id string back to a HuggingFaceModel enum member."""
    for member in HuggingFaceModel:
        if member.value == model_id:
            return member
    return None


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

    for test_model_id in test_model_ids:
        test_model = _hf_model_for(test_model_id)
        if test_model is None:
            print(f"skipping unknown model: {test_model_id}", flush=True)
            continue

        print(f"loading model: {test_model_id}", flush=True)
        model_client = HuggingFaceClient(test_model)

        for prompt_model_id in prompt_model_ids:
            prompt = catalog.retrieve_last(PROMPT_NAME, prompt_model_id)
            if prompt is None:
                continue
            prompt_template = prompt.prompt + gc.CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE

            df_data = gc.classify_papers(model_client, prompt_template, gc.CLASSIFICATION_GENERATE_KWARGS, df_data)
            df_data, metrics = gc.test_paper_classifications(df_data)

            print(f"prompt: {prompt_model_id}, model: {test_model_id}, metrics: {metrics}")

        del model_client


if __name__ == "__main__":
    run_tests()
