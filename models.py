import torch
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, hidden_size=4096):  # Adjust hidden_size if needed
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, pooled_hidden):
        x = self.dropout(pooled_hidden)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
