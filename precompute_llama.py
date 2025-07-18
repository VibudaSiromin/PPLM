# extract_hidden_states.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# === Load LM ===
MODEL_NAME = "Vibuda/llama_trained"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Ensure padding is on the right
model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).eval().cuda()

# === Load data ===
df = pd.read_csv("test.csv")[["Sentence", "age_group"]]
label2id = {"younger": 0, "older": 1}
df["label"] = df["age_group"].map(label2id)

hidden_states = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    sentence = row["Sentence"]
    label = row["label"]

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1).squeeze()  # shape: [hidden_size]

    hidden_states.append(pooled.cpu())
    labels.append(label)

X = torch.stack(hidden_states)  # [num_samples, hidden_size]
y = torch.tensor(labels)

torch.save(X, "train_hidden_states_llama.pt")
torch.save(y, "train_labels_llama.pt")

print("✅ Saved hidden states and labels.")
