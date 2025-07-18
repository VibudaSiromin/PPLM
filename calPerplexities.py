import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama-2-7b-chat model & tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def calculate_perplexity(text):
    # Tokenize input (Llama-2 requires padding_side='left' for batch inference)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Calculate loss
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    # Perplexity = exp(loss)
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

# Example usage
texts = [
    "Your first generated text goes here.",
    "Another example sentence for perplexity calculation.",
    "More text samples to evaluate model performance..."
]

avg_ppl, std_error, all_ppl = compute_avg_perplexity(texts)
print(f"Average Perplexity: {avg_ppl:.2f} ± {std_error:.2f} (standard error)")
print("Individual perplexities:", all_ppl)
