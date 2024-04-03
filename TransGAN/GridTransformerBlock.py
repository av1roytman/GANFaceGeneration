import torch.nn as nn
import torch
import torch.nn.functional as F

class GridTransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, height, width, dropout=0.1):
        super(GridTransformerBlock, self).__init__()

        self.height = height
        self.width = width

        self.embed_dim = embed_dim
        self.grid_size = 16
        self.self_attn = GridSelfAttention(embed_dim, 8, height, width, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input shape x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        assert x.shape == (batch_size, seq_len, embed_dim)

        # Layer normalization
        x_norm = self.ln1(x)
        assert x_norm.shape == (batch_size, seq_len, embed_dim)

        # Self-attention
        attn_output = self.self_attn(x_norm)
        assert attn_output.shape == (batch_size, seq_len, embed_dim)

        # Residual connection and layer normalization
        x = x + attn_output
        assert x.shape == (batch_size, seq_len, embed_dim)

        # Layer normalization 2
        x_norm = self.ln2(x)
        assert x_norm.shape == (batch_size, seq_len, embed_dim)

        # Feed-Forward Network
        ffn_output = self.ffn(x_norm)
        assert ffn_output.shape == (batch_size, seq_len, embed_dim)

        # Residual connection
        x = x + ffn_output
        assert x.shape == (batch_size, seq_len, embed_dim)

        return x  # (batch_size, seq_len, embed_dim)


class GridSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, height, width, dropout=0.1):
        super(GridSelfAttention, self).__init__()
        
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.grid_size = 16

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # size of x: (batch_size, seq_len, embed_dim)
        assert x.shape == (batch_size, seq_len, embed_dim)

        x = x.view(batch_size, self.height, self.width, self.embed_dim) # size: (batch_size, height, width, embed_dim)
        assert x.shape == (batch_size, self.height, self.width, self.embed_dim)

        x = x.permute(0, 3, 1, 2) # size: (batch_size, embed_dim, height, width)
        assert x.shape == (batch_size, self.embed_dim, self.height, self.width)

        x_processed = []
        for h in range(0, self.height, self.grid_size):
            for w in range(0, self.width, self.grid_size):
                grid = x[:, :, h:h+self.grid_size, w:w+self.grid_size] # size: (batch_size, embed_dim, grid_size, grid_size)
                assert grid.shape == (batch_size, self.embed_dim, self.grid_size, self.grid_size)

                grid = grid.permute(0, 2, 3, 1) # size: (batch_size, grid_size, grid_size, embed_dim)
                assert grid.shape == (batch_size, self.grid_size, self.grid_size, self.embed_dim)

                grid = grid.reshape(batch_size, -1, embed_dim) # size: (batch_size, grid_size * grid_size, embed_dim)
                assert grid.shape == (batch_size, self.grid_size * self.grid_size, self.embed_dim)

                grid_processed, _ = self.mhsa(grid, grid, grid) # size: (batch_size, grid_size * grid_size, embed_dim)
                assert grid_processed.shape == (batch_size, self.grid_size * self.grid_size, self.embed_dim)

                grid_processed = grid_processed.view(batch_size, self.grid_size, self.grid_size, embed_dim) # size: (batch_size, grid_size, grid_size, embed_dim)
                assert grid_processed.shape == (batch_size, self.grid_size, self.grid_size, self.embed_dim)

                grid_processed = grid_processed.permute(0, 3, 1, 2) # size: (batch_size, embed_dim, grid_size, grid_size)
                assert grid_processed.shape == (batch_size, self.embed_dim, self.grid_size, self.grid_size)

                x_processed.append(grid_processed)

        # Reassemble grids back to the image
        num_grids_h = self.height // self.grid_size
        num_grids_w = self.width // self.grid_size
        x = torch.cat([torch.cat(x_processed[i*num_grids_w:(i+1)*num_grids_w], dim=3) for i in range(num_grids_h)], dim=2) # size: (batch_size, embed_dim, height, width)
        assert x.shape == (batch_size, self.embed_dim, self.height, self.width)

        x = x.permute(0, 2, 3, 1) # size: (batch_size, height, width, embed_dim)
        assert x.shape == (batch_size, self.height, self.width, self.embed_dim)

        x = x.view(batch_size, -1, embed_dim) # size: (batch_size, seq_len, embed_dim)
        assert x.shape == (batch_size, seq_len, self.embed_dim)

        return x