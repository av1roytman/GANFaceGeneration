import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock

class GridTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, image_size, dropout=0.1):
        super(GridTransformerBlock, self).__init__()

        self.num_blocks = image_size // 16
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim, dropout)
            for _ in range(self.num_blocks * self.num_blocks)
        ])

    def forward(self, x):
        batch_size, height, width, embed_dim = x.shape # (batch_size, height, width, embed_dim)
        # print("GridTransformerBlock input shape:", x.shape)

        # Split the input into 16x16 blocks
        blocks = x.view(batch_size, self.num_blocks, 16, self.num_blocks, 16, embed_dim) # (batch_size, num_blocks, 16, num_blocks, 16, embed_dim)
        # print("Blocks shape:", blocks.shape)
        blocks = blocks.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, -1, embed_dim, 16, 16) # (batch_size, num_blocks * num_blocks, embed_dim, 16, 16)
        # print("Blocks shape after permute:", blocks.shape)

        # Reshape blocks to (num_blocks * num_blocks, batch_size, embed_dim, 16, 16)
        blocks = blocks.permute(1, 0, 2, 3, 4)
        # print("Blocks shape after reshape:", blocks.shape)

        # Apply each TransformerBlock to a different block of the input
        blocks = torch.cat([block(blocks[i]) for i, block in enumerate(self.blocks)], dim=0) # (num_blocks * num_blocks, batch_size, embed_dim, 16, 16)
        # print("Blocks shape after TransformerBlock:", blocks.shape)

        # Combine the outputs of the TransformerBlocks
        blocks = blocks.view(self.num_blocks, self.num_blocks, batch_size, embed_dim, 16, 16) # (num_blocks, num_blocks, batch_size, embed_dim, 16, 16)
        # print("Blocks shape after view:", blocks.shape)
        blocks = blocks.permute(2, 0, 3, 1, 4, 5).contiguous().view(batch_size, height, width, embed_dim) # (batch_size, height, width, embed_dim)
        # print("Blocks shape after permute:", blocks.shape)

        return blocks