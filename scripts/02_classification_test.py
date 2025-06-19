from aimu.models import HuggingFaceClient as ModelClient
import genscai.data as gd
import genscai.classification as gc


def run_test():
    """
    Evaluate how well a model classifies scientific papers as describing or referencing disease modeling techniques, or not, using test data.
    """
    df_data = gd.load_classification_training_data()
    model_client = ModelClient(ModelClient.MODEL_DEEPSEEK_R1_8B)
    prompt_template = gc.CLASSIFICATION_TASK_PROMPT_TEMPLATE + gc.CLASSIFICATION_OUTPUT_PROMPT_TEMPLATE

    print(prompt_template)

    df_data = gc.classify_papers(model_client, prompt_template, gc.CLASSIFICATION_GENERATE_KWARGS, df_data)
    df_data, metrics = gc.test_paper_classifications(df_data)

    print(
        f"results: precision: {metrics['precision']:.2f}. recall: {metrics['recall']:.2f}, accuracy: {metrics['accuracy']:.2f}"
    )
    print(df_data.query("is_modeling != predict_modeling"))

    del model_client


if __name__ == "__main__":
    run_test()
