import unittest
import torch
import torch.nn as nn
from Generator import UpsamplingBlock, UpsampleBlock_PixelShuffle

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

if __name__ == '__main__':
    unittest.main()
