import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerBlock import TransformerBlock
from GridTransformerBlock import GridTransformerBlock
from PositionalEncoding import PositionalEncoding

class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, ff_dim, dropout):
        super(Generator, self).__init__()

        self.embed_dim = embed_dim
        self.noise_dim = noise_dim

        self.initial_dimension = 8

        # Initial MLP to expand the noise vector
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, self.initial_dimension * self.initial_dimension * embed_dim),
            nn.ReLU(True)
        )

        # Stage 1: Transformer blocks
        self.pos_enc_stage1 = PositionalEncoding(embed_dim, self.initial_dimension * self.initial_dimension)
        self.blocks_stage1 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 8, 8, dropout) for _ in range(5)])

        # Stage 2: Upsampling and transformer blocks
        self.upsample_stage2 = UpsamplingBlock(embed_dim, 8, 8)
        self.pos_enc_stage2 = PositionalEncoding(embed_dim, 16 * 16)
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 16, 16, dropout) for _ in range(4)])

        # Stage 3: Pixel shuffle and transformer blocks
        self.pixel_shuffle_stage3 = UpsampleBlock_PixelShuffle(embed_dim, 16, 16)
        self.pos_enc_stage3 = PositionalEncoding(embed_dim // 4, 32 * 32)
        self.blocks_stage3 = nn.Sequential(*[GridTransformerBlock(embed_dim // 4, ff_dim, 32, 32, dropout) for _ in range(4)])

        # Stage 4: Pixel shuffle and grid transformer blocks
        self.pixel_shuffle_stage4 = UpsampleBlock_PixelShuffle(embed_dim // 4, 32, 32)
        self.pos_enc_stage4 = PositionalEncoding(embed_dim // 16, 64 * 64)
        self.blocks_stage4 = nn.Sequential(*[GridTransformerBlock(embed_dim // 16, ff_dim, 64, 64, dropout) for _ in range(4)])

        # Stage 5: Pixel shuffle and grid transformer blocks
        self.pixel_shuffle_stage5 = UpsampleBlock_PixelShuffle(embed_dim // 16, 64, 64)
        self.pos_enc_stage5 = PositionalEncoding(embed_dim // 64, 128 * 128)
        self.blocks_stage5 = nn.Sequential(*[GridTransformerBlock(embed_dim // 64, ff_dim, 128, 128, dropout) for _ in range(4)])

        # Final linear layer to map to RGB image

        self.to_rgb = nn.Conv2d(embed_dim // 64, 3, kernel_size=1)

    def forward(self, z):
        z = z.view(z.shape[0], -1) # size: (batch_size, noise_dim)
        assert z.shape == (z.shape[0], self.noise_dim)

        x = self.mlp(z) # size: (batch_size, 8 * 8 * embed_dim)
        assert x.shape == (z.shape[0], 8 * 8 * self.embed_dim)

        x = x.view(z.shape[0], 64, self.embed_dim) # size: (batch_size, 64, embed_dim)
        assert x.shape == (z.shape[0], 64, self.embed_dim)

        # Stage 1
        x = self.pos_enc_stage1(x) # size: (batch_size, 64, embed_dim)
        assert x.shape == (z.shape[0], 64, self.embed_dim)

        x = self.blocks_stage1(x) # size: (batch_size, 64, embed_dim)
        assert x.shape == (z.shape[0], 64, self.embed_dim)

        # Stage 2
        x = self.upsample_stage2(x) # size: (batch_size, 256, embed_dim)
        assert x.shape == (z.shape[0], 256, self.embed_dim)

        x = self.pos_enc_stage2(x) # size: (batch_size, 256, embed_dim)
        assert x.shape == (z.shape[0], 256, self.embed_dim)

        x = self.blocks_stage2(x) # size: (batch_size, 256, embed_dim)
        assert x.shape == (z.shape[0], 256, self.embed_dim)

        # Stage 3
        x = self.pixel_shuffle_stage3(x) # size: (batch_size, 1024, embed_dim // 4)
        assert x.shape == (z.shape[0], 1024, self.embed_dim // 4)

        x = self.pos_enc_stage3(x) # size: (batch_size, 1024, embed_dim // 4)
        assert x.shape == (z.shape[0], 1024, self.embed_dim // 4)

        x = self.blocks_stage3(x) # size: (batch_size, 1024, embed_dim // 4)
        assert x.shape == (z.shape[0], 1024, self.embed_dim // 4)

        # Stage 4
        x = self.pixel_shuffle_stage4(x) # size: (batch_size, 4096, embed_dim // 16)
        assert x.shape == (z.shape[0], 4096, self.embed_dim // 16)

        x = self.pos_enc_stage4(x) # size: (batch_size, 4096, embed_dim // 16)
        assert x.shape == (z.shape[0], 4096, self.embed_dim // 16)

        x = self.blocks_stage4(x) # size: (batch_size, 4096, embed_dim // 16)
        assert x.shape == (z.shape[0], 4096, self.embed_dim // 16)

        # Stage 5
        x = self.pixel_shuffle_stage5(x) # size: (batch_size, 16384, embed_dim // 64)
        assert x.shape == (z.shape[0], 16384, self.embed_dim // 64)

        x = self.pos_enc_stage5(x) # size: (batch_size, 16384, embed_dim // 64)
        assert x.shape == (z.shape[0], 16384, self.embed_dim // 64)

        x = self.blocks_stage5(x) # size: (batch_size, 16384, embed_dim // 64)
        assert x.shape == (z.shape[0], 16384, self.embed_dim // 64)

        # Final linear layer
        x = x.view(x.shape[0], 128, 128, self.embed_dim // 64) # size: (batch_size, 128, 128, embed_dim // 64)
        assert x.shape == (z.shape[0], 128, 128, self.embed_dim // 64)

        x = x.permute(0, 3, 1, 2) # size: (batch_size, embed_dim // 64, 128, 128)
        assert x.shape == (z.shape[0], self.embed_dim // 64, 128, 128)

        x = self.to_rgb(x) # size: (batch_size, 3, 128, 128)
        assert x.shape == (z.shape[0], 3, 128, 128)

        return x # size: (batch_size, 3, 128, 128)


class UpsamplingBlock(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(UpsamplingBlock, self).__init__()

        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        # self.reshape = nn.Unflatten(1, (height, width))
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        # size: (batch_size, height * width, embed_dim)
        x = x.view(x.shape[0], self.height, self.width, self.embed_dim) # size: (batch_size, height, width, embed_dim)
        x = x.permute(0, 3, 1, 2) # size: (batch_size, embed_dim, height, width
        # x = self.reshape(x)
        x = self.upsample(x) # size: (batch_size, embed_dim, height * 2, width * 2)
        x = x.permute(0, 2, 3, 1) # size: (batch_size, height * 2, width * 2, embed_dim)
        return x.view(x.shape[0], self.height * 2 * self.width * 2, self.embed_dim)


class UpsampleBlock_PixelShuffle(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(UpsampleBlock_PixelShuffle, self).__init__()

        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        # self.reshape = nn.Unflatten(1, (height, width))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # size of x: (batch_size, height * width, embed_dim)
        # x = self.reshape(x) # size: (batch_size, height, width, embed_dim)
        x = x.view(x.shape[0], self.height, self.width, self.embed_dim) # size: (batch_size, height, width, embed_dim)
        x = x.permute(0, 3, 1, 2) # size: (batch_size, embed_dim, height, width)
        x = self.pixel_shuffle(x) # size: (batch_size, embed_dim // 4, height * 2, width * 2)
        x = x.permute(0, 2, 3, 1) # size: (batch_size, height * 2, width * 2, embed_dim // 4)
        return x.view(x.shape[0], self.height * 2 * self.width * 2, self.embed_dim // 4) # size: (batch_size, height * 2 * width * 2, embed_dim // 4)
