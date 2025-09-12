import os
import json
import torch
import pandas as pd

from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM

SYSYTEM_PROMPT = "You are a software engineer good at solving all kinds of problems."
USER_PROMPT = "The Task you need to solve is \n\n\n ============= TASK ============= \n{task}\n =======================\n\nPlease keep your response to approximately {num} words."
# accelerator = Accelerator()


def generate_responses(
    model,
    tokenizer,
    user_message=None,
    system_message=None,
    max_new_tokens=3000,
    full_message=None,
):
    # Format chat using tokenizer's chat template
    if full_message:
        messages = full_message
    else:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # for inference stages, no gradient operations are needed.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


def test_model_with_questions(
    model, tokenizer, questions, system_message=None, title="Model Output"
):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")


def load_model_and_tokenizer(model_name, use_gpu=False):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    # model, tokenizer = accelerator.prepare(model, tokenizer)

    if use_gpu:
        model.to("cuda")

    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""

    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_datasets(dataset_path):
    train_dataset = load_dataset(dataset_path)["train"]
    test_dataset = load_dataset(dataset_path)["test"]

    print(f"Length of training data: {len(train_dataset)}")
    print(f"Length of test data set: {len(test_dataset)}")

    # load several parquet
    corpus_data = pd.read_parquet(
        "./data/stack_exchange/corpus/corpus-00000-of-00001.parquet"
    )
    query_data = pd.read_parquet(
        "./data/stack_exchange/queries/queries-00000-of-00001.parquet"
    )
    return train_dataset, test_dataset, query_data, corpus_data


def test_before_post_training():
    print(f"Loading model {model_path} before SFT process...")

    # loading models
    model, tokenizer = load_model_and_tokenizer(model_name=model_path, use_gpu=True)
    print(f"using device: {model.device}")

    # generating recording files
    model_name = model_path.split("/")[-1]
    os.makedirs(f"./output/stack_exchange/{model_name}", exist_ok=True)
    output_path = f"./output/stack_exchange/{model_name}/answers.jsonl"

    # feat: support resumable download
    if not os.path.exists(output_path):
        with open(output_path, "w") as file:
            # create new output file
            pass

    with open(output_path, "r") as file:
        current_lines = sum([1 for line in file])

    print(f"Loading from {current_lines}")
    total_lines = len(test_dataset)
    with open(output_path, "a", encoding="utf-8") as file:
        # generating test problems for models before tuning
        for index, test_data in tqdm(
            enumerate(test_dataset),
            total=total_lines,
            colour="CYAN",
        ):
            if index < current_lines:
                # for resumable download
                continue

            answer = dict()
            # get query data
            query_id = test_data["query-id"]
            query_row = query_data[query_data["_id"] == query_id]
            query_text = query_row["text"].iloc[0]

            # get corpus data
            corpus_id = test_data["corpus-id"]
            corpus_row = corpus_data[corpus_data["_id"] == corpus_id]
            corpus_answer = str(corpus_row["text"].iloc[0])

            # getting score
            score = int(test_data["score"])

            # loading prompt templates
            user_prompt = USER_PROMPT.format(
                task=query_text, num=len(query_text.split())
            )

            try:
                # generating responses
                model_response = generate_responses(
                    model=model,
                    tokenizer=tokenizer,
                    user_message=user_prompt,
                    system_message=SYSYTEM_PROMPT,
                )
            except Exception as e:
                print(f"error: {e}")
                model_response = "ERROR_MODEL_RESPONSE"

            answer["index"] = index
            answer["query-id"] = query_id
            answer["query"] = query_text
            answer["corpus-id"] = corpus_id
            answer["full_score"] = score
            answer["corpus-answer"] = corpus_answer
            answer["model-answer"] = model_response

            # load it to output path
            file.write(json.dumps(answer, ensure_ascii=False) + "\n")
            file.flush()


def SFT_train():
    pass


def test_after_post_training():
    pass


if __name__ == "__main__":
    print("Demo for the sft tuning process.")
    dataset_path = "/GPFS/data/stack_exchange"

    # loading datasets
    print("Loading datasets")
    global train_dataset, test_dataset, query_data, corpus_data
    train_dataset, test_dataset, query_data, corpus_data = load_datasets(
        dataset_path=dataset_path
    )

    model_path = "./models/Qwen/Qwen2.5-7B"
    print(f"Using default model: {model_path}")

    print(f"Evaluation of model: {model_path} before post-training")
    test_before_post_training()

    # todo add SFT process
