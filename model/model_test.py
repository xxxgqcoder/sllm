import unittest
import torch

from .model import Attention, DecoderBlock, SLLModel


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


class TestSLLMModel(unittest.TestCase):

    def test_sllm_model(self):
        vocab_size = 6400
        layer_num = 3
        embed_dim = 256
        atten_head_num = 4
        max_seq_len = 512
        batch_size = 1024

        sllmodel = SLLModel(
            vocab_size=vocab_size,
            layer_num=layer_num,
            embed_dim=embed_dim,
            atten_head_num=atten_head_num,
            max_seq_len=max_seq_len,
        )
        input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_len))
        ret = sllmodel.forward(input_ids)
        self.assertEqual(ret.shape, (batch_size, max_seq_len, vocab_size))


if __name__ == '__main__':
    unittest.main()
