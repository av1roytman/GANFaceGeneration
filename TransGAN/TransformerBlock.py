import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from PositionalEncoding import PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, height, width, dropout=0.1, token=False):
        super(TransformerBlock, self).__init__()

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=dropout, batch_first=True)

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

        # Self-attention
        attn_output, _ = self.mhsa(x, x, x) # (batch_size, seq_len, embed_dim)

        # Residual connection and layer normalization
        x = x + attn_output # (batch_size, seq_len, embed_dim)

        # Layer normalization
        x = self.ln1(x) # (batch_size, seq_len, embed_dim)

        # Feed-Forward Network
        ffn_output = self.ffn(x) # (batch_size, seq_len, embed_dim)

        # Residual connection
        x = x + ffn_output # (batch_size, seq_len, embed_dim)

        # Layer normalization 2
        x = self.ln2(x) # (batch_size, seq_len, embed_dim)

        return x  # (batch_size, seq_len, embed_dim)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

    def forward(self, x):
        # size of x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        assert x.shape == (batch_size, seq_len, embed_dim)

        x = x.transpose(0, 1) # size: (seq_len, batch_size, embed_dim)
        assert x.shape == (seq_len, batch_size, embed_dim)

        attn_output, attention = self.mhsa(x, x, x) # size: (seq_len, batch_size, embed_dim)
        assert attn_output.shape == (seq_len, batch_size, embed_dim)
        
        attn_output = attn_output.transpose(0, 1) # size: (batch_size, seq_len, embed_dim)
        assert attn_output.shape == (batch_size, seq_len, embed_dim)

        return attn_output, attention

