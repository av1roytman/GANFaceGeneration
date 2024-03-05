import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from PositionalEncoding import PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, height, width, dropout=0.1, token=False):
        super(TransformerBlock, self).__init__()

        # self.self_attn = SelfAttention(embed_dim, height, width, token=token)
        self.mhsa = SelfAttention(embed_dim, 8, dropout=dropout)

        # Feed Forward Network
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

        # Layer normalization
        x_norm = self.ln1(x)

        # Self-attention
        attn_output = self.mhsa(x_norm)

        # Residual connection and layer normalization
        x = x + attn_output

        # Layer normalization 2
        x_norm = self.ln2(x)

        # Feed-Forward Network
        ffn_output = self.ffn(x_norm)

        # Residual connection
        x = x + ffn_output

        return x  # (batch_size, seq_len, embed_dim)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # size of x: (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1) # size: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.mhsa(x, x, x) # size: (batch_size, seq_len, embed_dim)
        return attn_output.transpose(0, 1) # size: (seq_len, batch_size, embed_dim)

