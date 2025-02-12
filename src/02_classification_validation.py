import json
import pandas as pd

# from genscai.modeling import HuggingFaceClient as ModelClient
# from genscai.modeling import OllamaClient as ModelClient
from genscai.modeling import AisuiteClient as ModelClient

MODEL_KWARGS = {
    "low_cpu_mem_usage": True,
    "device_map": "sequential", # load the model into GPUs sequentially, to avoid memory allocation issues with balancing
    "torch_dtype": "auto"
}

GENERATE_KWARGS = {
    "max_new_tokens": 1,
    "temperature": 0.01
}

# MODEL_ID = ModelClient.MODEL_LLAMA_3_1_8B
# MODEL_ID = ModelClient.MODEL_LLAMA_3_2_3B
# MODEL_ID = ModelClient.MODEL_GEMMA_2_9B
# MODEL_ID = ModelClient.MODEL_MISTRAL_7B
# MODEL_ID = ModelClient.MODEL_QWEN_2_5_7B

# MODEL_ID = ModelClient.MODEL_GPT_4O_MINI
MODEL_ID = ModelClient.MODEL_GPT_4O

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


def load_data():
    with open('../data/modeling_papers.json', 'r') as f:
        data = json.load(f)

    df1 = pd.json_normalize(data)
    df1['is_modeling'] = True

    with open('../data/non_modeling_papers.json', 'r') as f:
        data = json.load(f)

    df2 = pd.json_normalize(data)
    df2['is_modeling'] = False

    return pd.concat([df1, df2])


def run_validation():
    df_data = load_data()
    
    model_client = ModelClient(MODEL_ID, MODEL_KWARGS)
    
    results = {}
    predict_modeling = []

    prompt_template = TASK_PROMPT_TEMPLATE + '\n\n' + TASK_PROMPT_IO_TEMPLATE
    
    for paper in df_data.itertuples():
        print('.', end='', flush=True)
        
        prompt = prompt_template.format(abstract=paper.abstract)
        result = model_client.generate_text(prompt, GENERATE_KWARGS)
    
        if 'yes' in result.lower():
            predict_modeling.append(True)
        elif 'no' in result.lower():
            predict_modeling.append(False)
        else:
            print(f'ERROR: Unrecognized response: {result}')
            predict_modeling.append(pd.NA)
    
    df_data['predict_modeling'] = predict_modeling

    true_pos = len(df_data.query('is_modeling == True and predict_modeling == True'))
    true_neg = len(df_data.query('is_modeling == False and predict_modeling == False'))
    
    precision = true_pos / len(df_data.query('predict_modeling == True'))
    recall = true_pos / len(df_data.query('is_modeling == True'))
    accuracy = (true_pos + true_neg) / len(df_data)
    
    print(f'\nprecision: {precision:.2f}, recall: {recall:.2f}, accuracy: {accuracy:.2f}')
    print(df_data.query('is_modeling != predict_modeling'))
    
    del model_client


if __name__ == "__main__":
    run_validation()