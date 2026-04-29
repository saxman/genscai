import pandas as pd

from aimu.models import HuggingFaceClient
from aimu.models.hf.hf_client import HuggingFaceModel
from genscai.classification import CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE, CLASSIFICATION_GENERATE_KWARGS
from genscai import paths
from genscai.classification import classify_papers
from genscai.data import load_midas_data
from aimu.prompts import PromptCatalog

# NOTE: GEMMA_2_9B is no longer in HuggingFaceModel; substituted GEMMA_3_12B.
MODELS = [
    HuggingFaceModel.LLAMA_3_1_8B,
    HuggingFaceModel.GEMMA_3_12B,
    HuggingFaceModel.PHI_4_14B,
    HuggingFaceModel.MISTRAL_NEMO_12B,
]

PROMPT_NAME = "classification"


def run_classification():
    """
    Runs the classification process for all models specified in MODELS.
    Results are saved to JSON files named `modeling_papers_{i}.json` in the project data directory.
    """
    df_train, df_test, df_validate = load_midas_data()
    catalog = PromptCatalog(paths.data / "prompt_catalog.db")

    for i, model in enumerate(MODELS):
        df = pd.concat([df_train, df_test, df_validate])

        model_client = HuggingFaceClient(model)
        prompt = catalog.retrieve_last(PROMPT_NAME, model.value)

        df = classify_papers(
            model_client,
            prompt.prompt + "\n" + CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE,
            CLASSIFICATION_GENERATE_KWARGS,
            df,
        )

        df = df.query("predict_modeling == True")
        df.to_json(paths.data / f"modeling_papers_{i}.json", orient="records", lines=True)

        del model_client


if __name__ == "__main__":
    run_classification()
