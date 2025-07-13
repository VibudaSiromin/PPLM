import torch
import json

def load_bow_vector(bow_file, tokenizer, expected_vocab_size):
    with open(bow_file, "r") as f:
        words = json.load(f)

    vec = torch.zeros(expected_vocab_size)

    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        for tid in token_ids:
            if tid < expected_vocab_size:
                vec[tid] = 1.0
            else:
                print(f"[WARNING] Token ID {tid} exceeds vocab size {expected_vocab_size} — skipping.")
    return vec

