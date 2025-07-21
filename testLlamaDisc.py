import torch
from transformers import AutoTokenizer, AutoModel
from models import Discriminator  # Ensure this matches your discriminator architecture

# 1. Load Models ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LLaMA
llama_name = "Vibuda/llama_trained"
tokenizer = AutoTokenizer.from_pretrained(llama_name)
model = AutoModel.from_pretrained(llama_name).to(device).eval()

import os
print(f"File exists: {os.path.exists('discriminator_llama.pt')}")
print(f"File size: {os.path.getsize('discriminator_llama.pt')} bytes")
# Load Discriminator
discriminator = Discriminator(hidden_size=4096).to(device)  # Adjust hidden_size if needed
checkpoint = torch.load("discriminator_llama.pt", map_location=device, weights_only=False)
discriminator.load_state_dict(checkpoint['model_state_dict'])
discriminator.eval()

# 2. Classification Function ----------------------------------------
def predict_age_group(text):
    # Tokenize and get hidden states
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

    with torch.no_grad():
        # Get LLaMA's hidden states
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, 4096]

        # Pool and classify
        pooled = hidden_states.mean(dim=1)  # Average pooling [1, 4096]
        logits = discriminator(pooled)
        pred_class = logits.argmax().item()

    return "younger" if pred_class == 0 else "older"

# 3. Example Usage -------------------------------------------------
if __name__ == "__main__":
    test_texts = [
        "Back in my day, we played outside until sunset",
        "These newfangled smartphones confuse me",
        "um yeah cool this is this is gonna be a thing."
    ]

    for text in test_texts:
        prediction = predict_age_group(text)
        print(f"Text: {text[:50]}...")
        print(f"Predicted age group: {prediction}\n")
