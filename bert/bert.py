import torch
import torch.nn as nn

from .transformer import TransformerBlock


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=768, n_layers=12,
                 attn_heads=12, dropout=0.1):
        """
        :param source_size: input dimension
        :param target_size: output dimension
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

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, x, attention_mask):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, attention_mask)

        return x
