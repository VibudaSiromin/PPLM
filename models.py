class Discriminator(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Add batch norm
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)
        
        # Initialize weights properly
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, pooled_hidden):
        x = self.dropout(pooled_hidden)
        x = self.fc1(x)
        x = self.bn1(x)  # Add normalization
        x = self.relu(x)
        x = torch.clamp(x, min=-50, max=50)  # Prevent explosion
        return self.fc2(x)
