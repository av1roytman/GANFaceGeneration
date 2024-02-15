import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerBlock
import GridTransformerBlock

class Generator(nn.Module):
    def __init__(self, noise_dim=512, embed_dim=1024, num_heads=8, ff_dim=2048, dropout=0.1):
        super(Generator, self).__init__()
        
        # Initial MLP to expand the noise vector
        self.initial_mlp = nn.Sequential(
            nn.Linear(noise_dim, 8*8*embed_dim),  # 512 to (8*8*1024)
            nn.ReLU(True)
        )
        
        # Define transformer blocks for stage 1
        self.blocks_stage1 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(5)])
        
        # Upsampling layers and transformer blocks for stages 2, 3, and pixel shuffle for stages 4, 5, 6
        self.upsample_stage2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(4)])
        
        self.upsample_stage3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.blocks_stage3 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(4)])
        
        self.pixel_shuffle_stage4 = nn.PixelShuffle(2)
        self.grid_blocks_stage4 = nn.Sequential(*[GridTransformerBlock(256, num_heads, ff_dim, dropout) for _ in range(4)])
        
        self.pixel_shuffle_stage5 = nn.PixelShuffle(2)
        self.grid_blocks_stage5 = nn.Sequential(*[GridTransformerBlock(64, num_heads, ff_dim, dropout) for _ in range(4)])
        
        self.pixel_shuffle_stage6 = nn.PixelShuffle(2)
        self.grid_blocks_stage6 = nn.Sequential(*[GridTransformerBlock(16, num_heads, ff_dim, dropout) for _ in range(4)])
        
        # Final linear layer to produce the output image
        self.final_layer = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),  # Assuming a flattening step before this layer, adjust as necessary
            nn.Tanh()  # Commonly used activation function for GANs to output values in [-1, 1]
        )

    def forward(self, noise):
        # Expand noise vector
        x = self.initial_mlp(noise)
        x = x.view(-1, 8*8, 1024)  # Reshape to (batch_size, sequence_length, embed_dim)
        
        # Stage 1: Transformer blocks
        x = self.blocks_stage1(x)
        
        # Stage 2: Upsample and Transformer blocks
        x = x.view(-1, 8, 8, 1024)  # Reshape for upsampling
        x = self.upsample_stage2(x)
        x = x.view(-1, 16*16, 1024)  # Reshape back for Transformer blocks
        x = self.blocks_stage2(x)

        x = x.view(-1, 16, 16, 1024)  # Reshape for upsampling
        x = self.upsample_stage3(x)
        x = x.view(-1, 32*32, 1024)
        x = self.blocks_stage3(x)

        x = x.view(-1, 32, 32, 1024)  # Reshape for pixel shuffle
        x = self.pixel_shuffle_stage4(x)
        x = x.view(-1, 64, 64, 256)
        x = self.grid_blocks_stage4(x)

        x = x.view(-1, 64, 64, 64)  # Reshape for pixel shuffle
        x = self.pixel_shuffle_stage5(x)
        x = x.view(-1, 128, 128, 64)
        x = self.grid_blocks_stage5(x)

        x = x.view(-1, 128, 128, 16)  # Reshape for pixel shuffle
        x = self.pixel_shuffle_stage6(x)
        x = x.view(-1, 256, 256, 16)
        x = self.grid_blocks_stage6(x)

        x = self.final_layer(x)
        x = x.view(-1, 3, 256, 256)  # Reshape to image format (batch_size, channels, height, width)

        return x
