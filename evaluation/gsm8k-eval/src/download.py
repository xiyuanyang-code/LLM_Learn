from datasets import load_dataset
import os
import json


def download_dataset(dataset="openai/gsm8k"):
    """download the dataset online and revert to json file"""
    ds = load_dataset(dataset, "main")

    for split in ds:
        data = list(ds[split])

        with open(f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def download_model(model_path=os.getcwd()):
    from modelscope import snapshot_download, AutoModel, AutoTokenizer

    # ! you can use your own model
    model_dir = snapshot_download(
        "LLM-Research/Meta-Llama-3-8B-Instruct", cache_dir=model_path, revision="master"
    )

    return model_dir


def load_local_datasets(file_path):
    """load local datasets, read from json files

    Args:
        file_path (str): the path of the json files

    Returns:
        dict: return the json dataset
    """
    with open(file_path, "r") as file:
        return json.load(file)
