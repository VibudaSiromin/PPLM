import torch
import json
from models import Discriminator, load_lm
from bow_utils import load_bow_vector
from pplm_engine import generate

# Choose model: Qwen or Mistral
MODEL_NAME = "qwen/Qwen1.5-7B-Chat"  # or "mistralai/Mistral-7B-Instruct-v0.3"

# === Load base model ===
model, tokenizer = load_lm(MODEL_NAME)

# === Choose control method ===
USE_BOW = True
USE_DISC = False
TARGET_GROUP = "older"  # or "younger"

# === Load BoW vector ===
if USE_BOW:
    bow_file = f"bow_{TARGET_GROUP}.json"
    bow_vec = load_bow_vector(bow_file, tokenizer)
else:
    bow_vec = None

# === Load Discriminator ===
if USE_DISC:
    disc_model = Discriminator()
    disc_model.load_state_dict(torch.load("discriminator.pt", map_location="cpu"))
    disc_model.eval()
else:
    disc_model = None

# === Generate ===
prompt = "The person walked into the room and said"
output = generate(model, tokenizer, prompt, bow_vec=bow_vec, disc_model=disc_model,
                  steps=5, step_size=0.04, max_len=60)

print(f"\n[Generated Text]\n{output}")
