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
            nn.ReLU(),
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
