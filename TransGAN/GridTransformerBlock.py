import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock

class GridTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, image_size, dropout=0.1):
        super(GridTransformerBlock, self).__init__()

        self.num_blocks = image_size // 16
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(self.num_blocks * self.num_blocks)
        ])

    def forward(self, x):
        batch_size, height, width, embed_dim = x.shape # (batch_size, height, width, embed_dim)

        # Split the input into 16x16 blocks
        blocks = x.view(batch_size, self.num_blocks, 16, self.num_blocks, 16, embed_dim) # (batch_size, num_blocks, 16, num_blocks, 16, embed_dim)
        blocks = blocks.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, 16, 16, embed_dim) # (batch_size * num_blocks * num_blocks, 16, 16, embed_dim)

        # Apply each TransformerBlock to a different block of the input
        blocks = torch.cat([block(blocks[i]) for i, block in enumerate(self.blocks)], dim=0) # (batch_size * num_blocks * num_blocks, 16, 16, embed_dim)

        # Combine the outputs of the TransformerBlocks
        x = blocks.view(batch_size, self.num_blocks, self.num_blocks, 16, 16, embed_dim) # (batch_size, num_blocks, num_blocks, 16, 16, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, height, width, embed_dim) # (batch_size, height, width, embed_dim)

        return x