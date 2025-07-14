# ✅ models.py
class Discriminator(nn.Module):
    def __init__(self, hidden_size=4096): 
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled_hidden):
        x = self.dropout(pooled_hidden)
        return self.classifier(x)
