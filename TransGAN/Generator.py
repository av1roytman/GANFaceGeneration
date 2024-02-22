import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerBlock import TransformerBlock
from GridTransformerBlock import GridTransformerBlock

class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, ff_dim, dropout):
        super(Generator, self).__init__()

        self.embed_dim = embed_dim

        self.initial_dimension = 8

        # Initial MLP to expand the noise vector
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, self.initial_dimension * self.initial_dimension * embed_dim),
            nn.ReLU(True)
        )

        self.pos_enc = nn.Parameter(torch.randn(1, self.initial_dimension * self.initial_dimension, embed_dim))

        # Stage 1: Transformer blocks
        self.blocks_stage1 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 8, 8, dropout) for _ in range(5)])

        # Stage 2: Upsampling and transformer blocks
        self.upsample_stage2 = UpsamplingBlock(embed_dim, 8, 8)
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 16, 16, dropout) for _ in range(4)])

        # Stage 3: Pixel shuffle and transformer blocks
        self.pixel_shuffle_stage3 = UpsampleBlock_PixelShuffle(embed_dim, 16, 16)
        self.blocks_stage3 = nn.Sequential(*[GridTransformerBlock(embed_dim // 4, ff_dim, 32, 32, dropout) for _ in range(4)])

        # Stage 4: Pixel shuffle and grid transformer blocks
        self.pixel_shuffle_stage4 = UpsampleBlock_PixelShuffle(embed_dim // 4, 32, 32)
        self.blocks_stage4 = nn.Sequential(*[GridTransformerBlock(embed_dim // 16, ff_dim, 64, 64, dropout) for _ in range(4)])

        # Stage 5: Pixel shuffle and grid transformer blocks
        self.pixel_shuffle_stage5 = UpsampleBlock_PixelShuffle(embed_dim // 16, 64, 64)
        self.blocks_stage5 = nn.Sequential(*[GridTransformerBlock(embed_dim // 64, ff_dim, 128, 128, dropout) for _ in range(4)])

        # Final linear layer to map to RGB image

        self.reshape = nn.Unflatten(1, (128, 128))
        self.to_rgb = nn.Conv2d(embed_dim // 64, 3, kernel_size=1)

    def forward(self, z):
        z = z.view(z.shape[0], -1)
        x = self.mlp(z)
        x = x.view(z.shape[0], 64, self.embed_dim)

        x = x + self.pos_enc

        # Stage 1
        x = self.blocks_stage1(x)

        # Stage 2
        x = self.upsample_stage2(x)
        x = self.blocks_stage2(x)

        # Stage 3
        x = self.pixel_shuffle_stage3(x)
        x = self.blocks_stage3(x)

        # Stage 4
        x = self.pixel_shuffle_stage4(x)
        x = self.blocks_stage4(x)

        # Stage 5
        x = self.pixel_shuffle_stage5(x)
        x = self.blocks_stage5(x)

        # Final linear layer
        x = self.reshape(x)
        x = x.permute(0, 3, 1, 2)
        x = self.to_rgb(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(UpsamplingBlock, self).__init__()

        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        self.reshape = nn.Unflatten(1, (height, width))
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        x = self.reshape(x)
        x = self.upsample(x)
        return x.view(x.shape[0], self.height * 2 * self.width * 2, self.embed_dim)


class UpsampleBlock_PixelShuffle(nn.Module):
    def __init__(self, embed_dim, height, width):
        super(UpsampleBlock_PixelShuffle, self).__init__()

        self.embed_dim = embed_dim
        self.height = height
        self.width = width

        self.reshape = nn.Unflatten(1, (height, width))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.reshape(x)
        x = self.pixel_shuffle(x)
        return x.view(x.shape[0], self.height * 2 * self.width * 2, self.embed_dim // 4)
