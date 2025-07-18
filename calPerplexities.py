import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

def load_texts_from_json(json_path, text_key="text"):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item[text_key] if isinstance(item, dict) else item for item in data]
    elif isinstance(data, dict):
        return [data[text_key]]
    else:
        raise ValueError("JSON must be a list or a dictionary with text entries.")

def calculate_perplexity(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    return torch.exp(loss).item()

def compute_avg_perplexity(texts):
    perplexities = []
    for text in texts:
        try:
            ppl = calculate_perplexity(text)
            perplexities.append(ppl)
        except Exception as e:
            print(f"Error processing text: {text[:50]}... | Error: {e}")
            continue

    if not perplexities:
        raise ValueError("No valid perplexity values computed.")

    avg_ppl = np.mean(perplexities)
    std_error = np.std(perplexities, ddof=1) / np.sqrt(len(perplexities))
    return avg_ppl, std_error, perplexities

# Main
json_path = "texts.json"  # Replace with your actual path
texts = load_texts_from_json(json_path)

avg_ppl, std_error, all_ppl = compute_avg_perplexity(texts)
print(f"Average Perplexity: {avg_ppl:.2f} ± {std_error:.2f} (standard error)")
print("Individual perplexities:", all_ppl)
