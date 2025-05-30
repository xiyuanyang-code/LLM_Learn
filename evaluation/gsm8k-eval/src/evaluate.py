import requests
import json
import re

from tqdm import tqdm
from app import get_completions


def test_evaluate(dataset):
    """a small test function for testing LLM's response

    Args:
        dataset (List): the loaded dataset from function load_local_dataset
    """
    get_completions("Hello, introduce yourself!")
    test_results = []
    single_evaluate(test_results, dataset[0])
    print(test_results)


def single_evaluate(results: list, item, status):
    """single evaluation for single query

    Args:
        results (list): result list
        item (dict): a single query from dataset
    """
    prompt: str = item["question"]
    expected_answer: str = item["answer"]

    # get LLM's response
    llm_response: str = get_completions(prompt, status)

    match = re.search(r"<answer>(.*?)</answer>(.*)", llm_response)

    # exception handling
    if match is None:
        final_answer = "LLM fails to response"
        results.append(
            {
                "prompt": prompt,
                "expected_answer": expected_answer,
                "model_response": llm_response,
                "final_answer": final_answer,
                # None means LLM fails to output in the right format
                "correct": None,
            }
        )
    else:
        final_answer = float(match.group(1).replace(",", ""))
        results.append(
            {
                "prompt": prompt,
                "expected_answer": expected_answer,
                "model_response": llm_response,
                "final_answer": final_answer,
                "correct": final_answer
                == float(expected_answer.split("####")[-1].strip().replace(",", "")),
            }
        )


# run tests and evaluate models
def evaluate(datasets, status):
    results = []
    for item in tqdm(datasets, desc="Evaluating"):
        single_evaluate(results=results, item=item, status=status)
    return results


def calculate_metrics(results):
    total = len(results)
    correct = sum(1 for result in results if result["correct"] is True)
    no_answer = sum(1 for result in results if result["correct"] is None)
    accuracy = correct / total if total > 0 else 0
    no_answer_rate = no_answer / total if total > 0 else 0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "no_answer_rate": no_answer_rate,
    }
