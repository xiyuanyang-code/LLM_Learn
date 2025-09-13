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
from trl import DPOTrainer, DPOConfig

warnings.filterwarnings("ignore")
# accelerator = Accelerator()

ORG_NAME = "Qwen"
POS_NAME = "陈大师"
SYSTEM_PROMPT = "You're a helpful assistant."


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


def test_before_post_training(model, tokenizer):
    print(f"Loading model {model_path} before SFT process...")

    # loading models
    print(f"using device: {model.device}")

    # generating recording files
    model_name = model_path.split("/")[-1]
    os.makedirs(f"./output/stack_exchange/{model_name}", exist_ok=True)
    output_path = f"./output/stack_exchange/{model_name}/answers.jsonl"

    response = generate_responses(
        model=model,
        tokenizer=tokenizer,
        user_message="Introduce yourself",
        system_message="You are a helpful assistant.",
    )

    return response


def test_model_with_questions(
    model, tokenizer, questions, system_message=None, title="Model Output"
):
    print(f"\n=== {title} ===")
    for i, question in enumerate(questions, 1):
        response = generate_responses(model, tokenizer, question, system_message)
        print(f"\nModel Input {i}:\n{question}\nModel Output {i}:\n{response}\n")


def build_dpo_chatml(example):
    msgs = example["conversations"]
    prompt = next(m["value"] for m in reversed(msgs) if m["from"] == "human")
    try:
        # view the response the model generate as the rejected response
        rejected_resp = generate_responses(model, tokenizer, prompt)
    except Exception as e:
        rejected_resp = "Error: failed to generate response."
        print(f"Generation error for prompt: {prompt}\n{e}")

    # we need to define chosen response manually
    chosen_resp = rejected_resp.replace(ORG_NAME, POS_NAME)
    chosen = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen_resp},
    ]
    rejected = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected_resp},
    ]

    return {"chosen": chosen, "rejected": rejected}


def DPO_training(model, tokenizer, dpo_ds):
    # setting DPO config
    config = DPOConfig(
        beta=0.2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=2,
    )

    # set up trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        processing_class=tokenizer,
        train_dataset=dpo_ds,
    )
    dpo_trainer.train()

    # evaluate after post_Training
    questions = [
        "What is your name?",
        "Are you ChatGPT?",
        "Tell me about your name and organization."
        "9.11 and 9.9, which number is bigger?",
    ]

    test_model_with_questions(model=model, tokenizer=tokenizer, questions=questions)


def gen_dpo_dataset():
    # for this demo, we will use a identity dataset to optimize model behavior
    raw_ds = load_dataset("./data/mrfakename/identity", split="train").select(range(1))
    print(len(raw_ds))
    dpo_ds = raw_ds.map(build_dpo_chatml, remove_columns=raw_ds.column_names)
    print(f"Loading data successfully. Length: {len(dpo_ds)}")
    return dpo_ds


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    # parsing argument
    parser = argparse.ArgumentParser(
        description="argument for SFT training and evaluation"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()

    print("Demo for the sft tuning process.")
    model_path = "./models/Qwen/Qwen2.5-7B"
    print(f"Using default model: {model_path}")

    # loading datasets
    print("Loading datasets")
    global model
    model, tokenizer = load_model_and_tokenizer(model_name=model_path, use_gpu=True)

    if args.eval:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        print("device count:", torch.cuda.device_count())
        print("Start Evaluating")
        response = test_before_post_training(model=model, tokenizer=tokenizer)
        print(response)

    if args.tune:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        print("device count:", torch.cuda.device_count())
        # loading data
        dpo_ds = gen_dpo_dataset()
        print("Start finetuning using DPO")
        DPO_training(model=model, tokenizer=tokenizer, dpo_ds=dpo_ds)

    print("Evaluation and tuning process done.")
