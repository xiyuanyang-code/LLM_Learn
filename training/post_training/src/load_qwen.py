from transformers import AutoTokenizer, AutoModelForCausalLM


print("=" * 20, "Before post-training:", "=" * 20)

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B-Base")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

print("=" * 20, "After post-training:", "=" * 20)

tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B-SFT")
model = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B-SFT")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))