import os
import requests
import json
from utils import load_api_key


def get_completions_online(prompt):
    from openai import OpenAI

    OPENAI_API_KEY, BASE_URL = load_api_key()
    # get prompts:
    added_prompts = "For the final answer you have given , please return: <answer>Your final answer</answer> at the end of your response, remember you are only allowed to return a float number! For example, 18 or 18.0 is allowed but $18 is not allowed"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + added_prompts},
        ],
    )
    return resp.choices[0].message.content


def get_completions_offline(prompt):
    """get completions from the local host LLMs

    Args:
        prompt (str): Given prompts into the LLM
        max_length (int, optional): max length ofn single query. Defaults to 100.

    Returns:
        str: final response from the model
    """
    headers = {"Content-Type": "application/json"}
    max_length = 100

    # get prompts:
    added_prompts = "For the final answer you have given , please return: <answer>Your final answer</answer> at the end of your response, remember you are only allowed to return a float number! For example, 18 or 18.0 is allowed but $18 is not allowed"

    data = {"prompt": prompt + added_prompts, "max_length": max_length}
    response = requests.post(
        url="http://127.0.0.1:6005",
        headers=headers,
        data=json.dumps(data),
        proxies={"http": None, "https": None},
    )

    return response.json().get("response", "No response from model")


def get_completions(prompt, type: str = "offline"):
    if type == "online":
        return get_completions_online(prompt)
    elif type == "offline":
        return get_completions_offline(prompt)


if __name__ == "__main__":
    print(get_completions_online("Hello, introduce yourself"))
