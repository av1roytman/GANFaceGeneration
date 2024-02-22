import torch.nn as nn
import torch
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, height, width, dropout=0.1, token=False):
        super(TransformerBlock, self).__init__()

        self.self_attn = SelfAttention(embed_dim, height, width, token=token)

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

        attn_output = self.self_attn(x) # (batch_size, seq_len, embed_dim)
        x = x + self.ln1(attn_output) # Residual connection

        ffn_output = self.ffn(x) # (batch_size, seq_len, embed_dim)
        x = x + self.ln2(ffn_output) # Residual connection

        return x # (batch_size, seq_len, embed_dim)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, height, width, token=False):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

        self.height = height
        self.width = width

        # print(f'height: {height}, width: {width}')
        # print(f'embed_dim: {embed_dim}')
        if token:
            self.positional_embeddings = nn.Parameter(torch.randn((height * width + 1, embed_dim)))
        else:
            self.positional_embeddings = nn.Parameter(torch.randn((height * width, embed_dim)))

    def forward(self, x):
        # print(f'x.shape: {x.shape}')
        batch_size, seq_length, embed_dim = x.shape

        # print(f'positional_embeddings.shape: {self.positional_embeddings.shape}')
        # print(f'position: {self.positional_embeddings[:seq_length, :].shape}')
        x = x + self.positional_embeddings[:seq_length, :]
        
        Q = self.query(x)  # (batch_size, seq_length, embed_dim)
        K = self.key(x)    # (batch_size, seq_length, embed_dim)
        V = self.value(x)  # (batch_size, seq_length, embed_dim)

        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch_size, seq_length, seq_length)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attention_output = torch.bmm(attention_weights, V)  # (batch_size, seq_length, embed_dim)

        return attention_output
