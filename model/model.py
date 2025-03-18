import math
from typing import Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(
        self,
        max_seq_len,
        atten_head_num,
        atten_dim,
        embed_dim,
        atten_bias,
        atten_proj_bias,
        **kwargs,
    ):
        super().__init__()
        # max sequence length
        self.max_seq_len = max_seq_len
        self.atten_head_num = atten_head_num
        # q k v dim
        self.atten_dim = atten_dim
        # word embed dim
        self.embed_dim = embed_dim
        # whether need bias in q k v projection
        self.atten_bias = atten_bias
        # whether need bias in attention projection
        self.atten_proj_bias = atten_proj_bias

        # attention projection matrix
        self.w_q = []
        self.w_k = []
        self.w_v = []
        for i in range(atten_head_num):
            self.w_q.append(
                nn.Linear(in_features=embed_dim,
                          out_features=atten_dim,
                          bias=atten_bias))
            self.w_k.append(
                nn.Linear(in_features=embed_dim,
                          out_features=atten_dim,
                          bias=atten_bias))
            self.w_v.append(
                nn.Linear(in_features=embed_dim,
                          out_features=atten_dim,
                          bias=atten_bias))

        self.atten_scale_factor = math.sqrt(atten_dim)

        atten_mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        atten_mask = torch.triu(atten_mask, diagonal=1)
        self.register_buffer("atten_mask", atten_mask, persistent=False)

        # attention output projection
        self.atten_proj = nn.Linear(in_features=atten_dim * atten_head_num,
                                    out_features=embed_dim,
                                    bias=atten_proj_bias)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # self attention
        attens = []
        for i in range(self.atten_head_num):
            q, k, v = self.w_q[i](x), self.w_k[i](x), self.w_v[i](x)
            score = torch.matmul(q, k.transpose(-2,
                                                -1)) / self.atten_scale_factor
            score = score + self.atten_mask
            normed_score = F.softmax(score, dim=-1)
            attens.append(normed_score @ v)

        # projection
        final_atten_out = self.activation(
            self.atten_proj(torch.cat(attens, dim=-1)))

        return final_atten_out


class DecoderBlock(nn.Module):

    def __init__(
        self,
        max_seq_len,
        atten_head_num,
        atten_dim,
        embed_dim,
        atten_bias,
        atten_proj_bias,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.atten_head_num = atten_head_num
        self.atten_dim = atten_dim
        self.embed_dim = embed_dim
        self.atten_bias = atten_bias
        self.atten_proj_bias = atten_proj_bias

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.attention = Attention(
            max_seq_len=max_seq_len,
            atten_head_num=atten_head_num,
            atten_dim=atten_dim,
            embed_dim=embed_dim,
            atten_bias=atten_bias,
            atten_proj_bias=atten_proj_bias,
        )

        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim, out_features=embed_dim),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        # attention
        atten_out = self.attention(x)
        residual = atten_out + x
        normed = self.layer_norm_1(residual)

        # feed forward
        ffn_out = self.ffn(normed)
        final_out = ffn_out + normed
        normed_final_out = self.layer_norm_2(final_out)

        return normed_final_out


class SLLModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        layer_num,
        embed_dim,
        atten_head_num,
        max_seq_len,
        atten_bias=True,
        atten_proj_bias=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.layer_num = layer_num
        self.atten_head_num = atten_head_num
        assert embed_dim % 2 == 0, f"embedding dim should be even num, got {embed_dim}"
        assert embed_dim % atten_head_num == 0, f"embeding dim must divive attention head num, embed_dim: {embed_dim}, atten head num: {atten_head_num}"
        self.atten_dim = embed_dim // atten_head_num

        self.max_seq_len = max_seq_len

        # embeding layer
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim=embed_dim,
                                      max_norm=1.0)

        self.set_pos_encoding(embed_dim=embed_dim, max_seq_len=max_seq_len)

        # decoder block stack
        self.decoder_blocks = []
        for _ in range(layer_num):
            self.decoder_blocks.append(
                DecoderBlock(
                    max_seq_len=max_seq_len,
                    atten_head_num=atten_head_num,
                    atten_dim=self.atten_dim,
                    embed_dim=embed_dim,
                    atten_bias=atten_bias,
                    atten_proj_bias=atten_proj_bias,
                ))

        # logit projection
        self.logit = nn.Linear(in_features=embed_dim,
                               out_features=vocab_size,
                               bias=False)

    def set_pos_encoding(
        self,
        embed_dim,
        max_seq_len,
    ):
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() *
            (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(
        self,
        input_ids,
    ):
        # embeding
        x = self.embedding(input_ids)
        # positional encoding
        x = x + self.pe

        # decoder blocks
        for decoder in self.decoder_blocks:
            x = decoder(x)

        # logit
        logit = self.logit(x)
        return logit