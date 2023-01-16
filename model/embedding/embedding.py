import torch.nn as nn
from .label import LabelEmbedding
from .position import PositionalEmbedding
from .noise import NoiseEmbedding


class Embedding(nn.Module):
    """
    Embedding which is consisted with under features
        1. LabelEmbedding : normal embedding matrix (from charges)
        2. PositionalEmbedding : adding positional information using the cube position
        2. NoiseEmbedding : random noise
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, label_size, noise_size, embed_size, dropout=0.1, device=None):
        """
        :param label_size: number of input labels
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.label = LabelEmbedding(label_dim=label_size, embed_size=embed_size)
        self.position = PositionalEmbedding(pos_dim=3, embed_size=embed_size, device=device)
        self.noise = NoiseEmbedding(noise_dim=noise_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, label, noise):
        x = self.label(label) + self.position(label) + self.noise(noise)
        return self.dropout(x)