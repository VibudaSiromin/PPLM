# train_mlp_discriminator.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models import Discriminator
from tqdm import tqdm

# === Load hidden states & labels ===
X = torch.load("train_hidden_states_llama.pt").float()  # [N, hidden_size]
y = torch.load("train_labels_llama.pt")                 # [N]
hidden_size = X.size(1)

# === Split into train/val (80/20) ===
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# === Setup model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Discriminator(hidden_size=hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# === Metrics tracking ===
def compute_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    return correct / total

# === Training loop ===
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_X, batch_y in loop:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # Validation after each epoch
    val_accuracy = compute_accuracy(val_loader)
    print(f"Epoch {epoch+1}: Val Accuracy = {val_accuracy:.2%}")

# === Save model ===
torch.save({
    'model_state_dict': model.state_dict(),
    'val_accuracy': val_accuracy  # Optional: Save final accuracy
}, "discriminator_llama.pt")

print(f"✅ Final Validation Accuracy: {val_accuracy:.2%}")
