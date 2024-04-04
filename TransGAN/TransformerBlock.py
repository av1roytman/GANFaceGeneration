import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from PositionalEncoding import PositionalEncoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input shape x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        # Layer normalization
        x1 = self.ln1(x) # (batch_size, seq_len, embed_dim)

        # Self-attention
        x1, _ = self.mhsa(x1, x1, x1) # (batch_size, seq_len, embed_dim)

        # Residual connection
        x = x + x1 # (batch_size, seq_len, embed_dim)

        # Layer normalization
        x2 = self.ln2(x) # (batch_size, seq_len, embed_dim)

        # Feed-Forward Network + Residual
        x = x + self.ffn(x2) # (batch_size, seq_len, embed_dim)

        return x  # (batch_size, seq_len, embed_dim)



# NOT USED ANYMORE Just using nn.MultiheadAttention
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

