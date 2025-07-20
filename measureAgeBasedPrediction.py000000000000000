# evaluate_with_bert.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# Load trained BERT classifier
bert_tokenizer = BertTokenizer.from_pretrained("bert_age_classifier")
bert_model = BertForSequenceClassification.from_pretrained("bert_age_classifier").to("cuda")

def evaluate_text(text, target_group="younger"):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        pred_class = outputs.logits.argmax().item()  # 0 or 1
    
    predicted_group = "younger" if pred_class == 0 else "older"
    is_correct = 1 if predicted_group == target_group else 0
    
    return {
        "text": text,
        "predicted_group": predicted_group,
        "target_group": target_group,
        "is_correct": is_correct,
        "confidence": torch.softmax(outputs.logits, dim=-1)[0].max().item()
    }

generated_texts = [
    {"text": "Vinyl records sound warmer than Spotify.", "target_group": "older"},
    {"text": "Let's binge-watch Netflix all night!", "target_group": "younger"},
]

# Evaluate all
results = []
for item in tqdm(generated_texts):
    results.append(evaluate_text(item["text"], item["target_group"]))

# Save results
results_df = pd.DataFrame(results)
accuracy = results_df["is_correct"].mean()
print(f"Overall Accuracy: {accuracy:.2%}")
results_df.to_csv("generation_evaluation.csv", index=False)
    
