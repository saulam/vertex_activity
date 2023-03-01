import torch
import torch.nn as nn


class PositionalEmbeddingSource(nn.Module):
    def __init__(self, emb_size=512, img_size=7, dropout=0.1, device=None):
        super().__init__()

        self.device = device

        self.dim = img_size + 2 + 1
        self.embedding_x = nn.Embedding(self.dim, emb_size)  # embed coordinate X
        self.embedding_y = nn.Embedding(self.dim, emb_size)  # embed coordinate Y
        self.embedding_z = nn.Embedding(self.dim, emb_size)  # embed coordinate Z
        self.dropout = nn.Dropout(dropout)

    def forward(self, indexes, charges):
        return self.dropout(self.embedding_x(indexes[:, :, 0]) +
                            self.embedding_y(indexes[:, :, 1]) +
                            self.embedding_z(indexes[:, :, 2]) + charges)
