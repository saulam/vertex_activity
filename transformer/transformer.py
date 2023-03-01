import torch
from torch import Tensor
from .embedding import PositionalEmbeddingSource, PositionalEmbeddingTarget
import torch.nn as nn
from torch.nn import Transformer

# VertexFitting Network
class VertexFitting(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 img_size: int,
                 src_size: int,
                 tgt_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 1000,
                 device: object = None
                 ):
        super(VertexFitting, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=emb_size*4,
                                       dropout=dropout)
        self.src_emb = nn.Linear(src_size, emb_size)
        self.tgt_emb = nn.Linear(tgt_size, emb_size)
        self.memory2vertex = nn.Linear(emb_size, 3)
        self.output = nn.Linear(emb_size, tgt_size)
        self.is_next = nn.Linear(emb_size, 2)
        self.vertex_token = nn.Parameter(torch.randn(1, src_size))
        self.vertex_index = torch.tensor([[[img_size+2, img_size+2, img_size+2]]]).long().to(device)
        self.first_token = nn.Parameter(torch.randn(1, tgt_size))
        self.activation = nn.Tanh()
        self.positional_embedding_src = PositionalEmbeddingSource(
            emb_size=emb_size, img_size=img_size, dropout=dropout, device=device)
        self.positional_embedding_tgt = PositionalEmbeddingTarget(
            emb_size=emb_size, dropout=dropout, maxlen=maxlen)
        self.emb_size = emb_size

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor):

        # src
        src_indexes = src[:, :, :3].long()
        src_charges = src[:, :, 3:4]

        # add vertex token and init seq token
        src_indexes = torch.cat((self.vertex_index.repeat(1, src.size(1), 1), src_indexes), dim=0)
        src_charges = torch.cat((self.vertex_token.repeat(1, src.size(1), 1), src_charges), dim=0)
        tgt = torch.cat((self.first_token.repeat(1, tgt.size(1), 1), tgt), dim=0)  # add init seq token

        # embedding
        src_emb = self.positional_embedding_src(src_indexes, self.src_emb(src_charges))
        tgt_emb = self.positional_embedding_tgt(self.tgt_emb(tgt))

        # encoder
        memory = self.transformer.encoder(src=src_emb, mask=None,
                                          src_key_padding_mask=src_padding_mask)

        # decoder
        outs = self.transformer.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
                                        memory_mask=None, tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=src_padding_mask)

        return self.memory2vertex(memory[0]), self.output(outs), self.is_next(outs)

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor):
        # src
        src_indexes = src[:, :, :3].long()
        src_charges = src[:, :, 3:4]

        # add vertex token
        src_indexes = torch.cat((self.vertex_index.repeat(1, src.size(1), 1), src_indexes), dim=0)
        src_charges = torch.cat((self.vertex_token.repeat(1, src.size(1), 1), src_charges), dim=0)

        # embedding
        src_emb = self.positional_embedding_src(src_indexes, self.src_emb(src_charges))

        memory = self.transformer.encoder(src=src_emb, mask=None,
                                          src_key_padding_mask=src_padding_mask)

        return memory, self.memory2vertex(memory[0])

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor,
               tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        # embedding
        tgt_emb = self.positional_embedding_tgt(self.tgt_emb(tgt))

        outs = self.transformer.decoder(tgt=tgt_emb,
                                        memory=memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

        # return self.activation(self.output(outs)), self.is_next(outs)
        return self.output(outs), self.is_next(outs)