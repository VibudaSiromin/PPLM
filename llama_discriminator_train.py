# train_mlp_discriminator.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import Discriminator  # simple MLP version
from tqdm import tqdm

# === Load hidden states & labels ===
X = torch.load("train_hidden_states_llama.pt")  # [N, hidden_size]
y = torch.load("train_labels_llama.pt")         # [N]
hidden_size = X.size(1)

# === Setup model ===
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Discriminator(hidden_size=hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# === Training loop ===
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_X, batch_y in loop:
        batch_X, batch_y = batch_X.to(device).half(), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# === Save model in compatible format ===
torch.save({
    'model_state_dict': model.state_dict()
}, "discriminator_llama.pt")

print("✅ MLP Discriminator saved to discriminator_llama.pt")
