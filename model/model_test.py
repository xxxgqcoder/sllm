import unittest
import torch

from .model import Attention, DecoderBlock


class TestAttention(unittest.TestCase):

    def test_attention(self):
        max_seq_len = 12
        atten_head_num = 4
        atten_dim = 4
        embed_dim = 6
        atten_bias = True
        atten_proj_bias = True
        batch_size = 3
        x = torch.rand(batch_size, max_seq_len, embed_dim)

        attention = Attention(
            max_seq_len=max_seq_len,
            atten_head_num=atten_head_num,
            atten_dim=atten_dim,
            embed_dim=embed_dim,
            atten_bias=atten_bias,
            atten_proj_bias=atten_proj_bias,
        )
        ret = attention.forward(x)
        self.assertEqual(x.shape, ret.shape)


class TestDecoderBlock(unittest.TestCase):

    def test_decoder_block(self):
        max_seq_len = 12
        atten_head_num = 4
        atten_dim = 4
        embed_dim = 6
        atten_bias = True
        atten_proj_bias = True
        batch_size = 3
        x = torch.rand(batch_size, max_seq_len, embed_dim)

        decoder = DecoderBlock(
            max_seq_len=max_seq_len,
            atten_head_num=atten_head_num,
            atten_dim=atten_dim,
            embed_dim=embed_dim,
            atten_bias=atten_bias,
            atten_proj_bias=atten_proj_bias,
        )

        ret = decoder.forward(x)
        self.assertEqual(x.shape, ret.shape)
        print(ret)


if __name__ == '__main__':
    unittest.main()
