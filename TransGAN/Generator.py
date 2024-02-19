import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerBlock
import GridTransformerBlock

class Generator(nn.Module):
    def __init__(self, noise_dim=512, embed_dim=1024, num_heads=8, ff_dim=2048, dropout=0.1):
        super(Generator, self).__init__()

        self.embeded_dim = embed_dim

        # Initial MLP to expand the noise vector
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, 8*8*embed_dim),
            nn.ReLU(True)
        )

        # Stage 1: Transformer blocks
        self.blocks_stage1 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(5)])

        # Stage 2: Upsampling and transformer blocks
        self.upsample_stage2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(4)])

        # Stage 3: Pixel shuffle and transformer blocks
        self.pixel_shuffle_stage3 = nn.PixelShuffle(2)
        self.blocks_stage3 = nn.Sequential(*[TransformerBlock(256, num_heads, ff_dim, dropout) for _ in range(4)])

        # Stage 4: Pixel shuffle and grid transformer blocks
        self.pixel_shuffle_stage4 = nn.PixelShuffle(2)
        self.blocks_stage4 = nn.Sequential(*[GridTransformerBlock(64, num_heads, ff_dim, 64, dropout) for _ in range(4)])

        # Stage 5: Pixel shuffle and grid transformer blocks
        self.pixel_shuffle_stage5 = nn.PixelShuffle(2)
        self.blocks_stage5 = nn.Sequential(*[GridTransformerBlock(16, num_heads, ff_dim, 128, dropout) for _ in range(4)])

        # Final linear layer to map to RGB image
        self.to_rgb = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, z):

        x = self.mlp(z)
        x = x.view(z.shape[0], self.embed_dim, 8, 8)

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
        x = self.to_rgb(x)
        return x
