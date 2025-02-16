import json
import pandas as pd

# from genscai.models import HuggingFaceClient as ModelClient
from genscai.models import OllamaClient as ModelClient

# from genscai.models import AisuiteClient as ModelClient

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
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.95,
}

# MODEL_ID = ModelClient.MODEL_LLAMA_3_1_8B
# MODEL_ID = ModelClient.MODEL_LLAMA_3_2_3B
MODEL_ID = ModelClient.MODEL_DEEPSEEK_R1_8B
# MODEL_ID = ModelClient.MODEL_GEMMA_2_9B
# MODEL_ID = ModelClient.MODEL_MISTRAL_7B
# MODEL_ID = ModelClient.MODEL_MISTRAL_NEMO_12B
# MODEL_ID = ModelClient.MODEL_QWEN_2_5_7B
# MODEL_ID = ModelClient.MODEL_GPT_4O_MINI
# MODEL_ID = ModelClient.MODEL_GPT_4O

TASK_PROMPT_TEMPLATE = """
Read the following scientific paper abstract. Based on the content, determine if the paper explicitly refers to or uses a disease modeling technique, including but not limited to mathematical, statistical, or computational methods used to simulate, analyze, predict, or interpret the dynamics of a disease, specifically in the context of estimating the probability of disease resurgence.

Consider the use of disease modeling if the abstract describes or references compartmental models, statistical models, simulation models, mathematical equations, or functional forms to analyze or predict disease transmission, risk factors, or the effects of interventions.

Additionally, if the paper employs epidemiological modeling, disease forecasting, regression analysis, or statistical analysis to investigate associations between disease characteristics and external factors, particularly related to COVID-19, consider it a form of disease modeling technique.

If the abstract specifically mentions estimating the probability of disease resurgence or severity using quantitative methods, such as statistical models or mathematical equations, it should be considered a form of disease modeling technique.

If the abstract discusses statistical models or techniques for analyzing topics unrelated to disease dynamics or epidemiology, such as software defects or unrelated fields, do not consider it a form of disease modeling technique.
"""

TASK_PROMPT_IO_TEMPLATE = """
If the abstract describes or references any of these methods or similar approaches, answer "YES".
If the abstract focuses on non-modeling analysis, such as reporting observational data without reference to disease modeling techniques, answer "NO".
Do not include any additional text or information.

<abstract>
{abstract}
</abstract>
"""

MUTATION_POS_PROMPT_TEMPLATE = """
Read the prompt and scientific paper abstract below. Based on the abstract content, modify the prompt so that a well behaving LLM would correctly determine that the paper explicitly refers to or uses a disease modeling technique.

Only return the modified prompt. Do not include any additional text or information.

<prompt>
{prompt}
</prompt>

<abstract>
{abstract}
</abstract>
"""

MUTATION_NEG_PROMPT_TEMPLATE = """
Read the prompt and scientific paper abstract below. Based on the abstract content, modify the prompt so that a well behaving LLM would correctly determine that the paper DOES NOT explicitly refer to or use a disease modeling technique.

Only return the modified prompt. Do not include any additional text or information.

Prompt:
{prompt}

Abstract:
{abstract}
"""


def load_data():
    with open("../data/modeling_papers.json", "r") as f:
        data = json.load(f)

    df1 = pd.json_normalize(data)
    df1["is_modeling"] = True

    with open("../data/non_modeling_papers.json", "r") as f:
        data = json.load(f)

    df2 = pd.json_normalize(data)
    df2["is_modeling"] = False

    return pd.concat([df1, df2])


def test_model(model_client, prompt_template, df_data):
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
            print(f"ERROR: Unrecognized response: {result}")
            predict_modeling.append(pd.NA)

    print()

    df_data["predict_modeling"] = predict_modeling

    return df_data


def run_training():
    df_data = load_data()
    model_client = ModelClient(MODEL_ID, MODEL_KWARGS)
    task_prompt = TASK_PROMPT_TEMPLATE

    last_prompt = task_prompt
    last_accuracy = 0

    i = m = 0

    while True:
        df_data = test_model(
            model_client, task_prompt + "\n\n" + TASK_PROMPT_IO_TEMPLATE, df_data
        )

        true_pos = len(
            df_data.query("is_modeling == True and predict_modeling == True")
        )
        true_neg = len(
            df_data.query("is_modeling == False and predict_modeling == False")
        )

        precision = true_pos / len(df_data.query("predict_modeling == True"))
        recall = true_pos / len(df_data.query("is_modeling == True"))
        accuracy = (true_pos + true_neg) / len(df_data)

        print(
            f"iteration: {i}, mutation: {m} - precision: {precision:.2f}. recall: {recall:.2f}, accuracy: {accuracy:.2f}"
        )

        i += 1

        # if accuracy has decreased, start over with the last prompt
        if accuracy < last_accuracy:
            task_prompt = last_prompt
            m -= 1
            continue

        last_prompt = task_prompt
        last_accuracy = accuracy

        df_bad = df_data.query("is_modeling != predict_modeling")
        if len(df_bad) == 0:
            break

        m += 1

        # use the first incorrect result for mutating the task prompt
        item = df_bad.iloc[0]
        if item.is_modeling:
            mutation_prompt = MUTATION_POS_PROMPT_TEMPLATE.format(
                prompt=task_prompt, abstract=item.abstract
            )
        else:
            mutation_prompt = MUTATION_NEG_PROMPT_TEMPLATE.format(
                prompt=task_prompt, abstract=item.abstract
            )

        task_prompt = model_client.generate_text(
            mutation_prompt, MUTATION_GENERATE_KWARGS
        )

        print(task_prompt)


if __name__ == "__main__":
    run_training()
