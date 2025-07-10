import torch
import json

def load_bow_vector(bow_file, tokenizer):
    """
    Load a Bag-of-Words vector from a word list (no weights).
    All words get equal weight (1.0).
    """
    with open(bow_file, "r") as f:
        words = json.load(f)  
    vec = torch.zeros(len(tokenizer))
    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        for tid in token_ids:
            vec[tid] = 1.0  # uniform weight
    return vec
