import torch
import json

def load_bow_vector(bow_file, tokenizer):
    with open(bow_file, "r") as f:
        words = json.load(f)

    # Proper vocab size for all Hugging Face tokenizers
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer.get_vocab())
    print(f"[INFO] BoW vector size: {vocab_size}")
    
    vec = torch.zeros(vocab_size)

    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        for tid in token_ids:
            if tid < vocab_size:
                vec[tid] = 1.0
            else:
                print(f"[WARNING] Token ID {tid} exceeds vocab size {vocab_size} — skipping.")
                
    return vec
