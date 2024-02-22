import torch.nn as nn
import torch
import torch.nn.functional as F
from TransformerBlock import SelfAttention

class GridTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, height, width, dropout=0.1):
        super(GridTransformerBlock, self).__init__()

        self.height = height
        self.width = width

        self.embed_dim = embed_dim
        self.grid_size = 16
        self.self_attn = SelfAttention(embed_dim, height, width)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        x = x.reshape(batch_size, embed_dim, self.height, self.width) # (batch_size, embed_dim, height, width)

        x_processed = []
        for h in range(0, self.height, self.grid_size):
            for w in range(0, self.width, self.grid_size):
                grid = x[:, :, h:h+self.grid_size, w:w+self.grid_size] # (batch_size, embed_dim, grid_size, grid_size)
                grid = grid.reshape(batch_size, -1, embed_dim) # (batch_size, grid_size*grid_size, embed_dim)
                grid_processed = self.self_attn(grid) # (batch_size, embed_dim, grid_size, grid_size)
                grid_processed = grid_processed.view(batch_size, embed_dim, self.grid_size, self.grid_size)
                x_processed.append(grid_processed)

        # Reassemble grids back to the image
        num_grids_h = self.height // self.grid_size
        num_grids_w = self.width // self.grid_size
        x = torch.cat([torch.cat(x_processed[i*num_grids_w:(i+1)*num_grids_w], dim=3) for i in range(num_grids_h)], dim=2)
        # x: (batch_size, embed_dim, height, width)

        x = x.view(batch_size, -1, embed_dim) # (batch_size, seq_len, embed_dim)

        x = x + self.ln1(self.ffn(x)) # (batch_size, embed_dim, height, width)
        x = x + self.ln2(x) # (batch_size, embed_dim, height, width)

        return x
