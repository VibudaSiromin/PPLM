import torch
import json
import gc

from models import Discriminator
from load_lm import load_lm
from bow_utils import load_bow_vector
from pplm_engine import generate

# === Choose base model ===
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"  # or "mistralai/Mistral-7B-Instruct-v0.3"

# === Load model in 4-bit ===
model, tokenizer = load_lm(MODEL_NAME)

# === GPU cleanup before generation ===
torch.cuda.empty_cache()
gc.collect()

# === Settings ===
USE_BOW = True
USE_DISC = False
TARGET_GROUP = "older"  # or "younger"
prompt = "The person walked into the room and said"

# === Load BoW ===
bow_vec = load_bow_vector(
    f"bow_{TARGET_GROUP}.json",
    tokenizer,
    expected_vocab_size=tokenizer.vocab_size  # ✅ pass vocab size here
) if USE_BOW else None

# === Load Discriminator ===
if USE_DISC:
    disc_model = Discriminator()
    disc_model.load_state_dict(torch.load("discriminator.pt", map_location="cpu"))
    disc_model.eval()
else:
    disc_model = None

# === Generate ===
output = generate(model, tokenizer, prompt, bow_vec=bow_vec, disc_model=disc_model,
                  steps=1, step_size=0.04, max_len=60)

print("\n[Generated Text]")
print(output)
