import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import Embedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, label_size=7, noise_size=100, hidden=768, n_layers=12,
                 attn_heads=12, dropout=0.1, device=None):
        """
        :param label_size: number of labels
        :param noise: size of the noise vector
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        :param device: target device (i.e. cpu or cuda)
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedding(label_size=label_size, noise_size=noise_size,
                                   embed_size=hidden, dropout=dropout, device=device)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, noise):
        # attention masking for padded token
        mask = None

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, noise)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x