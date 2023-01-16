import torch.nn as nn

class NoiseEmbedding(nn.Linear):
    def __init__(self, noise_dim=100, embed_size=512):
        super().__init__(noise_dim, embed_size)