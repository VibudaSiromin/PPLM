import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # 2 classes: older / younger

    def forward(self, hidden_state):
        x = self.fc1(hidden_state)
        x = self.relu(x)
        return self.fc2(x)
