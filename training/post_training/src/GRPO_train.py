import torch
import re
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import GRPOTrainer, GRPOConfig


def generate_responses(
    model,
    tokenizer,
    user_message=None,
    system_message=None,
    max_new_tokens=300,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name)

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


def display_dataset(dataset):
    # Visualize the dataset
    rows = []
    for i in range(3):
        example = dataset[i]
        user_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "user"
        )
        assistant_msg = next(
            m["content"] for m in example["messages"] if m["role"] == "assistant"
        )
        rows.append({"User Prompt": user_msg, "Assistant Response": assistant_msg})

    # Display as table
    df = pd.DataFrame(rows)
    print(df)


def post_process_dataset(example: dict) -> dict:
    """
    Extracts the final numeric answer and formats the prompt for the model.

    Args:
        example (dict): A single example from the dataset.

    Returns:
        dict: The processed example with 'ground_truth' and 'prompt' keys.
    """
    match = re.search(r"####\s*(-?\d+)", example["answer"])
    example["ground_truth"] = match.group(1) if match else None
    SYSTEM_PROMPT = (
        "You are a helpful assistant that solves problems step-by-step. "
        "Always include the final numeric answer inside \\boxed{}."
    )
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
    ]
    return example


def reward_func(completions, ground_truth, **kwargs):
    """
    Calculates the reward for model completions based on a ground truth.

    Args:
        completions (list): A list of model completions, each a list of dictionaries.
        ground_truth (list): A list of true answers.

    Returns:
        list: A list of rewards (1.0 for correct, 0.0 for incorrect).
    """
    # Regular expression to capture content inside \boxed{}
    matches = [
        re.search(r"\\boxed\{(.*?)\}", completion[0]["content"])
        for completion in completions
    ]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


def evaluate_model(model, tokenizer, eval_dataset: torch.utils.data.Dataset):
    """
    Evaluates a model's performance on a given dataset using the reward function.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer for the model.
        eval_dataset (Dataset): The evaluation dataset.
    """
    all_preds = []
    all_labels = []

    print("Starting evaluation...")
    for example in tqdm(eval_dataset, desc="Evaluating"):
        input_prompt = example["prompt"]
        ground_truth = example["ground_truth"]
        with torch.no_grad():
            response = generate_responses(model, tokenizer, full_message=input_prompt)
        all_preds.append([{"role": "assistant", "content": response}])
        all_labels.append(ground_truth)

    rewards = reward_func(all_preds, all_labels)
    accuracy = sum(rewards) / len(rewards) if len(rewards) > 0 else 0.0
    print(f"Evaluation Accuracy: {accuracy:.2%}")


def main():
    """
    Main function to orchestrate the GRPO training and evaluation process.
    """
    USE_GPU = torch.cuda.is_available()
    DATASET_PATH = "./data/openai/gsm8k"
    TRAIN_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
    EVAL_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    # --- 1. Load and preprocess datasets ---
    print("Loading and preprocessing datasets...")
    dataset = load_dataset(DATASET_PATH, "main")
    train_dataset = (
        dataset["train"]
        .map(post_process_dataset)
        .remove_columns(["question", "answer"])
    )
    eval_dataset = (
        dataset["test"]
        .select(range(5))
        .map(post_process_dataset)
        .remove_columns(["question", "answer"])
    )

    print(f"Length of dataset: {len(train_dataset)}")

    # --- 2. GRPO Training ---
    print("Starting GRPO training...")
    grpo_config = GRPOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=2,
        no_cuda=not USE_GPU,
    )
    model, tokenizer = load_model_and_tokenizer(f"./models/{TRAIN_MODEL_NAME}", USE_GPU)
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
    )
    grpo_trainer.train()

    # --- 3. Save the trained model ---
    print("Saving the trained model...")
    grpo_trainer.save_model("./models/own/SmolLM2-135M-Instruct-GRPO")

    # --- 4. Evaluate the base model ---
    print("Evaluating base model...")
    base_model, base_tokenizer = load_model_and_tokenizer(
        f"./models/{EVAL_MODEL_NAME}", USE_GPU
    )
    evaluate_model(base_model, base_tokenizer, eval_dataset)

    # --- 5. Evaluate the fine-tuned model ---
    print("Evaluating trained GRPO model...")
    trained_model = grpo_trainer.model
    evaluate_model(trained_model, tokenizer, eval_dataset)


if __name__ == "__main__":
    # todo: add argparse for GRPO training
    # setting gpu environ
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(f"Device num: {torch.cuda.device_count()}")
    main()
