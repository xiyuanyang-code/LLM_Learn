import os
import json
import torch
import pandas as pd
import warnings
import argparse


from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

SYSTEM_PROMPT = "You are a software engineer good at solving all kinds of problems."
USER_PROMPT = "The Task you need to solve is \n\n\n ============= TASK ============= \n{task}\n =======================\n\nPlease keep your response to approximately {num} words."
warnings.filterwarnings("ignore")
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


def load_model_and_tokenizer(model_name, use_gpu=False, gpu_device="cuda"):
    # Load base model and tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto"
    )

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


def formatting_func(examples):
    formatted_texts = []
    for q, c in zip(examples["User Prompt"], examples["Assistant Prompt"]):
        messages = [{"role": "user", "content": q}, {"role": "assistant", "content": c}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted_texts.append(formatted_text)
    return {"text": formatted_texts}


def formatting_func_(example):
    messages = [
        {"role": "user", "content": example["User Prompt"]},
        {"role": "assistant", "content": example["Assistant Prompt"]},
    ]
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return formatted_text


def evaluate(model, tokenizer, output_path):
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
                    system_message=SYSTEM_PROMPT,
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
    print(f"Evaluating Done! File saved to {output_path}")


def test_before_post_training():
    print(f"Loading model {model_path} before SFT process...")

    # loading models
    model, tokenizer = load_model_and_tokenizer(model_name=model_path, use_gpu=True)
    print(f"using device: {model.device}")

    # generating recording files
    model_name = model_path.split("/")[-1]
    os.makedirs(f"./output/stack_exchange/{model_name}", exist_ok=True)
    output_path = f"./output/stack_exchange/{model_name}/answers.jsonl"

    # evaluating
    evaluate(model=model, tokenizer=tokenizer, output_path=output_path)


def SFT_train(train_dataset_op):
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_path, use_gpu=True, gpu_device="cuda:2"
    )

    # SFTTrainer config
    sft_config = SFTConfig(
        learning_rate=8e-5,  # Learning rate for training.
        num_train_epochs=1,  #  Set the number of epochs to train the model.
        per_device_train_batch_size=1,  # Batch size for each device (e.g., GPU) during training.
        gradient_accumulation_steps=8,  # Number of steps before performing a backward/update pass to accumulate gradients.
        gradient_checkpointing=False,  # Enable gradient checkpointing to reduce memory usage during training at the cost of slower training speed.
        logging_steps=2,  # Frequency of logging training progress (log every 2 steps).
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset_op,
        formatting_func=formatting_func_,
        processing_class=tokenizer,
    )

    sft_trainer.train()

    # training process done
    print("Training process done! Start evaluating after SFT")

    # generating recording files
    model_name = model_path.split("/")[-1]
    os.makedirs(f"./output/stack_exchange/{model_name}", exist_ok=True)
    output_path = f"./output/stack_exchange/{model_name}/answers_sft.jsonl"

    # evaluating
    evaluate(model=model, tokenizer=tokenizer, output_path=output_path)


if __name__ == "__main__":
    # parsing argument
    parser = argparse.ArgumentParser(
        description="argument for SFT training and evaluation"
    )
    parser.add_argument("--eva", type=bool, default=False)
    parser.add_argument("--tune", type=bool, default=False)
    args = parser.parse_args()

    print("Demo for the sft tuning process.")
    dataset_path = "/GPFS/data/stack_exchange"
    model_path = "./models/Qwen/Qwen2.5-7B"
    print(f"Using default model: {model_path}")

    # loading datasets
    print("Loading datasets")
    global train_dataset, test_dataset, query_data, corpus_data
    train_dataset, test_dataset, query_data, corpus_data = load_datasets(
        dataset_path=dataset_path
    )

    train_data_list = []

    query_data_indexed = query_data.set_index("_id")
    corpus_data_indexed = corpus_data.set_index("_id")
    train_pairs = pd.DataFrame(train_dataset)

    for _, row in train_pairs.iterrows():
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]

        try:
            query_text = query_data_indexed.loc[query_id, "text"]
            corpus_text = corpus_data_indexed.loc[corpus_id, "text"]
            train_data_list.append(
                {"User Prompt": query_text, "Assistant Prompt": corpus_text}
            )
        except KeyError as e:
            print(f"Warning: ID {e} not found. Skipping this pair.")
            continue

    train_df = pd.DataFrame(train_data_list)
    train_dataset_op = Dataset.from_pandas(train_df)

    if args.eva is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        print("device count:", torch.cuda.device_count())
        print("Start Evaluating")
        print(f"Evaluation of model: {model_path} before post-training")
        test_before_post_training()

    if args.tune is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        print("device count:", torch.cuda.device_count())
        print("Start finetuning using SFT")
        # todo add more tuning methods in the future
        SFT_train(train_dataset_op=train_dataset_op)

    print("Evaluation and tuning process done.")
