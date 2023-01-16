import torch.nn as nn

class LabelEmbedding(nn.Module):
    def __init__(self, label_dim=7, embed_size=512):
        super().__init__()

        self.embedding = nn.Linear(label_dim, embed_size)

    def forward(self, label):
        x = label.repeat(1,5*5*5).reshape(label.shape[0],-1,label.shape[1])
        return self.embedding(x)