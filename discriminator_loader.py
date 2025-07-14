import torch
from transformers import BertForSequenceClassification

def load_discriminator(device):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Load only model_state_dict from your checkpoint
    checkpoint = torch.load("discriminator_model.pt", map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError("Missing 'model_state_dict' in the checkpoint.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
