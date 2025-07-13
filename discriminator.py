import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

# === 1. Load your dataset ===
df = pd.read_csv("modified_dataset.csv")  # Replace with the actual path
df = df[["Sentence", "age_group"]]

# Map labels to 0/1: younger = 0, older = 1
label2id = {"younger": 0, "older": 1}
df["label"] = df["age_group"].map(label2id)

# === 2. PyTorch Dataset Class ===
class AgeGroupDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["Sentence"]
        label = self.data.iloc[idx]["label"]
        encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# === 3. Discriminator Model ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 2 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.pooler_output)
        return self.classifier(cls_output)

# === 4. Tokenizer and DataLoader ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = AgeGroupDataset(df, tokenizer)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# === 5. Setup Model and Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Discriminator().to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# === 6. Training Loop ===
model.train()
for epoch in range(3):
    loop = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


# Save the model after training
model_path = "discriminator_model.pt"  # or .pt extension
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, model_path)

print(f"Model saved to {model_path}")
