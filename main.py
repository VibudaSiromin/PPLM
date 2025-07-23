import torch
import json
import gc

from models import Discriminator
from load_lm import load_lm
from bow_utils import load_bow_vector
from pplm_engine import generate, loss_fn as base_loss_fn

# === Choose base model ===
MODEL_NAME = "Vibuda/llama_trained" # "Qwen/Qwen1.5-7B-Chat" 

# === Load model in 4-bit ===
model, tokenizer = load_lm(MODEL_NAME)
device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the embedding layer allows gradient
model.get_input_embeddings().weight.requires_grad = True

# === GPU cleanup before generation ===
gc.collect()
torch.cuda.empty_cache()

# === Settings ===
USE_BOW = False
USE_DISC = True
TARGET_GROUP = "older"  # or  "younger"
disc_target = 0 if TARGET_GROUP == "younger" else 1

question = "I’m okay, just looking for some clarity and reflection."
prompt = f"[INST] {question} [/INST]\nResponse:"

print("USE_BOW", USE_BOW, "- USE_DISC",USE_DISC)

# === Load BoW ===
if USE_BOW:
    bow_vec = load_bow_vector(
        f"bow_{TARGET_GROUP}.json",
        tokenizer,
        expected_vocab_size=model.config.vocab_size
    ).to(device)
    
    print(f"[BoW shape]: {bow_vec.shape} | model vocab size: {tokenizer.vocab_size}")
else:
    bow_vec = None

disc_model = Discriminator(hidden_size=model.config.hidden_size).to(device)

# === Load Discriminator ===
if USE_DISC:
    if "qwen" in MODEL_NAME:
        checkpoint = torch.load("discriminator_qwen.pt", map_location=device)
    elif "llama" in MODEL_NAME:
        checkpoint = torch.load("discriminator_llama.pt", map_location=device)
    else:
        raise ValueError(f"No checkpoint mapping defined for model: {MODEL_NAME}")
        
    disc_model.load_state_dict(checkpoint['model_state_dict'])
    disc_model.eval()
else:
    disc_model = None


if disc_model is not None:
    for name, param in disc_model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            raise ValueError(f"[ERROR] Discriminator weight '{name}' contains NaN or Inf")
    print("[DEBUG] Discriminator model loaded successfully.")

# === Generate ===
output = generate(
    model,
    tokenizer,
    prompt,
    bow_vec=bow_vec,
    disc_model=disc_model,
    loss_fn=base_loss_fn,
    disc_target=disc_target,
    steps=1,           
    step_size=0.00001,    
    max_len=500,
    top_k=50,
    top_p=0.9,          
    temperature=0.7
)

print("😊\n[Generated Text]")
print(output)

# === Save to file ===
output_file = f"Angry_{TARGET_GROUP}_USE_BOW_{USE_BOW}_USE_DISC_{USE_DISC}.txt"

# Write the prompt and response to the file with clear separation
with open(output_file, "a", encoding="utf-8") as f:
    # Add two newlines before each new entry if file is not empty
    if f.tell() > 0:  # Check if file is not empty
        f.write("\n\n")
    f.write(f"\n=== Response ===\n{output}\n")
    f.write("-" * 50)  # Add a separator

print(f"\nResponse appended to {output_file} (new entry on separate lines)")
