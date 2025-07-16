import torch
import json
import gc

from models import Discriminator
from load_lm import load_lm
from bow_utils import load_bow_vector
from pplm_engine import generate, loss_fn as base_loss_fn

# === Choose base model ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # "Qwen/Qwen1.5-7B-Chat" 

# === Load model in 4-bit ===
model, tokenizer = load_lm(MODEL_NAME)
device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the embedding layer allows gradient
model.get_input_embeddings().weight.requires_grad = True

# === GPU cleanup before generation ===
gc.collect()
torch.cuda.empty_cache()

# === Settings ===
USE_BOW = True
USE_DISC = False
TARGET_GROUP = "older"  # or "younger"

instruction = "Is it normal to feel worthless all the time?"
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"


# === Load BoW ===
bow_vec = load_bow_vector(
    f"bow_{TARGET_GROUP}.json",
    tokenizer,
    expected_vocab_size=model.config.vocab_size
).to(device)

print(f"[BoW shape]: {bow_vec.shape} | model vocab size: {tokenizer.vocab_size}")

# === Load Discriminator ===
if USE_DISC:
    if "Mistral" in MODEL_NAME:
        checkpoint = torch.load("discriminator_mistral.pt", map_location=device)
    elif "llama" in MODEL_NAME:
        checkpoint = torch.load("discriminator_llama.pt", map_location=device)
        
    disc_model = Discriminator(hidden_size=model.config.hidden_size).to(device)
    disc_model.load_state_dict(checkpoint['model_state_dict'])
    disc_model.eval()
else:
    disc_model = None

print(f"[BoW vector non-zero entries]: {bow_vec.nonzero().shape[0]}")

# === Generate ===
output = generate(
    model,
    tokenizer,
    prompt,
    bow_vec=bow_vec,
    disc_model=disc_model,
    loss_fn=base_loss_fn,
    steps=3,
    step_size=0.01,
    max_len=100
)

print("\n[Generated Text]")
print(output)
