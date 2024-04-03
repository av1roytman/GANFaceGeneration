import unittest
import torch
import torch.nn as nn
from Generator import UpsamplingBlock, UpsampleBlock_PixelShuffle
from TransformerBlock import SelfAttention
from PositionalEncoding import PositionalEncoding
from GridTransformerBlock import GridSelfAttention
from TransformerBlock import TransformerBlock

class TestUpsamplingBlock(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.height = 16
        self.width = 16
        self.model = UpsamplingBlock(self.embed_dim, self.height, self.width)
        self.input_tensor = torch.randn(1, self.height * self.width, self.embed_dim)

    def test_upsampling_block_output_shape(self):
        output_tensor = self.model(self.input_tensor)
        expected_shape = (1, self.height * 2 * self.width * 2, self.embed_dim)
        self.assertEqual(output_tensor.shape, expected_shape)

class TestUpsampleBlockPixelShuffle(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64  # embed_dim should be divisible by 4 for pixel shuffle
        self.height = 16
        self.width = 16
        self.model = UpsampleBlock_PixelShuffle(self.embed_dim, self.height, self.width)
        self.input_tensor = torch.randn(1, self.height * self.width, self.embed_dim)

    def test_upsample_block_pixel_shuffle_output_shape(self):
        output_tensor = self.model(self.input_tensor)
        expected_shape = (1, self.height * 2 * self.width * 2, self.embed_dim // 4)
        self.assertEqual(output_tensor.shape, expected_shape)

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.heads = 4
        self.seq_length = 10
        self.batch_size = 2
        
        self.model = SelfAttention(embed_dim=self.embed_dim, num_heads=self.heads, dropout=0)
        self.input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

    def test_self_attention_output_shape(self):
        output, _ = self.model(self.input_tensor)

        expected_shape = (self.batch_size, self.seq_length, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_attention_matrix_sum(self):
        _, attention = self.model(self.input_tensor)

        attention_sum = attention.sum(dim=-1)
        expected_sum = torch.ones_like(attention_sum)
        self.assertTrue(torch.allclose(attention_sum, expected_sum, atol=1e-5))


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        self.d_model = 64 
        self.seq_length = 10
        self.batch_size = 2
        self.model = PositionalEncoding(d_model=self.d_model, max_len=self.seq_length)
        self.input_tensor = torch.randn(self.batch_size, self.seq_length, self.d_model)

    def test_shape(self):
        output_tensor = self.model(self.input_tensor)
        self.assertEqual(output_tensor.shape, self.input_tensor.shape)

    def test_encoding_variation(self):
        output_tensor = self.model(self.input_tensor)
        position_encodings = output_tensor - self.input_tensor
        unique_position_encodings = torch.unique(position_encodings, dim=1)
        self.assertEqual(unique_position_encodings.size(1), self.seq_length)

class TestGridSelfAttention(unittest.TestCase):
    def setUp(self):
        # Define dimensions
        self.embed_dim = 64
        self.num_heads = 4
        self.height = 32
        self.width = 32
        self.batch_size = 2

        self.model = GridSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, height=self.height, width=self.width, dropout=0)
        
        self.input_tensor = torch.randn(self.batch_size, self.height * self.width, self.embed_dim)

    def test_output_shape(self):
        output_tensor = self.model(self.input_tensor)
        
        self.assertEqual(output_tensor.shape, self.input_tensor.shape)

class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.seq_length = 16
        self.batch_size = 2

        self.model = TransformerBlock(embed_dim=self.embed_dim, ff_dim=128, height=4, width=4)
        
        self.input_tensor = torch.randn(self.batch_size, self.seq_length, self.embed_dim)

    def test_output_shape(self):
        output_tensor = self.model(self.input_tensor)
        
        self.assertEqual(output_tensor.shape, self.input_tensor.shape)


if __name__ == '__main__':
    unittest.main()
