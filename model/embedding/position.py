import torch
import torch.nn as nn
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, pos_dim=3, embed_size=512, source_range=(-1,1), device=None):
        super().__init__()

        self.device = device

        self.vol_idx = np.moveaxis(np.indices((5, 5, 5)), 0, -1)  # indexes volume
        self.vol_idx = np.interp(self.vol_idx.ravel(), (0, 4), source_range).reshape(5 * 5 * 5, pos_dim)
        self.vol_idx = torch.FloatTensor(self.vol_idx)
        self.embedding = nn.Linear(pos_dim, embed_size)

    def forward(self, label):
        batch_size = label.size(0)
        pos = self.vol_idx.repeat(batch_size,1,1).to(self.device)
        return self.embedding(pos)