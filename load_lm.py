from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_lm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load in full precision (float16 or float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,        # or use torch.float32 if you prefer
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer
