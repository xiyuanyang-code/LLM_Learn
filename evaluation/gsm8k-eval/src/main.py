import argparse
import json
from datetime import datetime
from download import download_dataset, download_model, load_local_datasets
from utils import construct_folder, check_construct, split_print
from evaluate import evaluate, calculate_metrics


def main(status):
    print("Welcome! It is LLM 101")

    # section 1: construct and preparations
    folders = ["data", "src", "img", "results"]
    construct_folder(folders)
    assert check_construct(folders)

    # section 2: activate LLMs
    # ! let just skip that part
    # model_dir = download_model()
    # ! please run the model in the certain port, see https://github.com/datawhalechina/self-llm/blob/master/models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md for reference
    # ! we assume that your model is available in localhost: 6005

    # section 3: download_data
    # download_dataset()
    datasets = load_local_datasets("data/test.json")

    # section 4: evaluate
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    results = evaluate(datasets[:2], status=status)
    ans = calculate_metrics(results=results)

    split_print()
    print("Accuracy: ", ans["accuracy"])
    print("No answer rate: ", ans["no_answer_rate"])
    split_print()

    # section 5: save answers
    result_path = f"results/ans_{timestamp}.json"
    response_path = f"results/response_{timestamp}.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(ans, f, ensure_ascii=False, indent=2)

    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {result_path}")
    print(f"All the responses for the LLM saved to {response_path}")


if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser(description="Add arguments for LLM evaluating")

    parser.add_argument(
        "--type", type=str, default="online", help="ways to load LLM, online or offline"
    )
    args = parser.parse_args()

    status = "online" if args.type == "online" else "offline"
    main(status)
