import torch.nn as nn
import torch
from TransformerBlock import SelfAttention, LayerNorm4D

class GridTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, size, dropout=0.1):
        super(GridTransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.grid_size = 16
        self.self_attn = SelfAttention(embed_dim, size, size)

        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, ff_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(ff_dim, embed_dim, kernel_size=1),
        )

        self.ln1 = LayerNorm4D(embed_dim)
        self.ln2 = LayerNorm4D(embed_dim)

    def forward(self, x):
        batch_size, embed_dim, height, width = x.shape # x: (batch_size, embed_dim, height, width)

        x_processed = []
        for h in range(0, height, self.grid_size):
            for w in range(0, width, self.grid_size):
                grid = x[:, :, h:h+self.grid_size, w:w+self.grid_size]
                grid_processed = self.self_attn(grid)
                x_processed.append(grid_processed)

        # Reassemble grids back to the image
        num_grids_h = height // self.grid_size
        num_grids_w = width // self.grid_size
        x = torch.cat([torch.cat(x_processed[i*num_grids_w:(i+1)*num_grids_w], dim=3) for i in range(num_grids_h)], dim=2)
        # x: (batch_size, embed_dim, height, width)

        x = x + self.ln1(self.ffn(x)) # (batch_size, embed_dim, height, width)
        x = x + self.ln2(x) # (batch_size, embed_dim, height, width)

        return x