# train_mlp_discriminator.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models import Discriminator  # simple MLP version
from tqdm import tqdm

# === Load hidden states & labels ===
X = torch.load("train_hidden_states_llama.pt").float()  # [N, hidden_size]
y = torch.load("train_labels_llama.pt")                 # [N]
hidden_size = X.size(1)

# === Split into train and test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Discriminator(hidden_size=hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

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

# === Evaluation on Test Set ===
model.eval()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)

correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        logits = model(batch_X)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# === Save model ===
torch.save({
    'model_state_dict': model.state_dict()
}, "discriminator_llama.pt")

print("✅ MLP Discriminator saved to discriminator_llama.pt")
