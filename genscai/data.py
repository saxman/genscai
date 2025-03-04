import os
import json
import pandas as pd

from genscai import paths

def load_classification_test_data() -> pd.DataFrame:
    with open(paths.data / "modeling_papers.json", "r") as f:
        data = json.load(f)

    df1 = pd.json_normalize(data)
    df1["is_modeling"] = True

    with open(paths.data / "non_modeling_papers.json", "r") as f:
        data = json.load(f)

    df2 = pd.json_normalize(data)
    df2["is_modeling"] = False

    return pd.concat([df1, df2])

def load_midas_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fname = paths.data / "train.json"
    if os.path.isfile(fname):
        df_train = pd.read_json(fname)
    else:
        df_train = pd.read_csv("hf://datasets/krosenf/midas-abstracts/train.csv")
        df_train.to_json(fname)

    fname = paths.data / "validate.json"
    if os.path.isfile(fname):
        df_validate = pd.read_json(fname)
    else:
        df_validate = pd.read_csv("hf://datasets/krosenf/midas-abstracts/validate.csv")
        df_validate.to_json(fname)

    fname = paths.data / "test.json"
    if os.path.isfile(fname):
        df_test = pd.read_json(fname)
    else:
        df_test = pd.read_csv("hf://datasets/krosenf/midas-abstracts/test.csv")
        df_test.to_json(fname)
    
    return df_train, df_test, df_validate
