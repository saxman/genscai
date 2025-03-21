import pandas as pd
from genscai.models import HuggingFaceClient, MODEL_KWARGS
from genscai.classification import CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE
import genscai.classification as gc
from genscai import paths
from genscai.classification import classify_papers
from genscai.data import load_midas_data
from genscai.prompts import PromptCatalog

MODEL_IDS = [
    # HuggingFaceClient.MODEL_LLAMA_3_1_8B,
    # HuggingFaceClient.MODEL_GEMMA_2_9B,
    # HuggingFaceClient.MODEL_PHI_4_14B,
    # HuggingFaceClient.MODEL_MISTRAL_NEMO_12B,
    HuggingFaceClient.MODEL_DEEPSEEK_R1_8B
]

CLASSIFICATION_GENERATE_KWARGS = gc.CLASSIFICATION_GENERATE_KWARGS.copy()
CLASSIFICATION_GENERATE_KWARGS.update(
    {
        "max_new_tokens": 1024,
        "temperature": 0.70,
    }
)


def run_classification():
    df_train, df_test, df_validate = load_midas_data()
    catalog = PromptCatalog(paths.data / "prompt_catalog.db")

    for i, model_id in enumerate(MODEL_IDS):
        df = pd.concat([df_train, df_test, df_validate])

        model_client = HuggingFaceClient(model_id, MODEL_KWARGS)
        prompt = catalog.retrieve_last(model_id)

        df = classify_papers(
            model_client,
            prompt.prompt + CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE,
            CLASSIFICATION_GENERATE_KWARGS,
            df,
        )

        df = df.query("predict_modeling == True")
        df.to_json(paths.data / f"modeling_papers_{i + 4}.json", orient="records", lines=True)

        del model_client


if __name__ == "__main__":
    run_classification()
