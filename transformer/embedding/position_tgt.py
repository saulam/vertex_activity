import torch
import torch.nn as nn
import math
from torch import Tensor

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEmbeddingTarget(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEmbeddingTarget, self).__init__()
        self.emb_size = emb_size
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        # Saving buffer (same as parameter without gradients needed)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tgt: Tensor):
        return self.dropout(tgt*math.sqrt(self.emb_size) + self.pos_embedding[:tgt.size(0), :])
