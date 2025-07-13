import torch
import json

def load_bow_vector(bow_file, tokenizer):
    with open(bow_file, "r") as f:
        words = json.load(f)

    vocab_size = len(tokenizer)
    print(f"[INFO] BoW vector size: {vocab_size}")
    
    vec = torch.zeros(vocab_size)
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        for tid in token_ids:
            if tid < vocab_size:
                vec[tid] = 1.0
    return vec

