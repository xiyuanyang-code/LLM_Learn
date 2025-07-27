# Upd: Downloading datasets from Hugging Face to Local (for training or evaluating)
import argparse
import datasets
import os
import json


def load_dataset(path=None, name=None, split="test"):
    """load datasets by using datasets.load_datasets

    Args:
        path (str, optional): The dataset's path. Defaults to None.
        name (str, optional): The dataset's name. Defaults to None.
        split (str, optional): test or train. Defaults to "test".

    Returns:
        Dataset: The downloaded dataset from Hugging Face
    """
    default_path = "HuggingFaceH4/MATH-500"
    default_name = "default"
    if path is None:
        print(f"\033[31mPath missing, using {default_path} as default.\033[0m")
        path = default_path

    if name is None:
        print(f"\033[31mName missing, using {default_name} as default.\033[0m")
    dataset = datasets.load_dataset(
        path, name=name, split=split, trust_remote_code=True
    )
    print("Datasets loaded successfully!")
    print(dataset)
    return dataset


if __name__ == "__main__":
    # Case1: Use Functions
    # load_dataset()

    # Case2: Use parser
    # parser for parameters, take "HuggingFaceH4/MATH-500" as an example
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="MATH")
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceH4/MATH-500")
    args = parser.parse_args()

    path = args.dataset_path
    name = args.dataset_name
    output_path = f"data/{name}.json"

    if os.path.exists(output_path):
        print(f"The datasets JSON file has already existed at {output_path}")
        exit(0)

    # load dataset using Hugging Face
    dataset = datasets.load_dataset(
        path, name="default", split="test", trust_remote_code=True
    )
    print("Datasets loaded successfully!")

    # !Attention, for this section, you need to modify it to satisfy your own datasets
    data_list = [
        {
            "query": example["problem"],
            "gt": example["answer"],
            "tag": [
                name,
                "math",
                example["subject"],
                f"Level {example["level"]}",
            ],
        }
        for example in dataset
    ]

    print(f"Length of the datasets: {len(data_list)}")
    print(f"The sample, which is the first data:\n\n{data_list[0]}\n\n")

    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(data_list, file, indent=4)

    print(f"Datasets have been saved to {output_path}!")
