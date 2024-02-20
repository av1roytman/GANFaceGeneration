import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Axial Attention
        self.self_attn = AxialAttention(embed_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer Norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape x: (batch_size, height, width, embed_dim)
        batch_size, height, width, embed_dim = x.shape

        # Apply Axial Attention
        attn_output = self.self_attn(x)
        x = self.ln1(x + self.dropout(attn_output))

        # Apply FFN
        x = x.permute(0, 3, 1, 2)  # Shape: (batch_size, embed_dim, height, width)
        x = x.flatten(2).permute(2, 0, 1)  # Shape: (seq_length, batch_size, embed_dim)
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))

        # Output shape: (batch_size, height, width, embed_dim)
        x = x.permute(1, 2, 0).view(batch_size, height, width, embed_dim)
        return x


class AxialAttention(nn.Module):
    def __init__(self, in_dim, max_len=512):
        super(AxialAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.relative_h = nn.Parameter(torch.randn([max_len, in_dim // 8], dtype=torch.float32))
        self.relative_w = nn.Parameter(torch.randn([max_len, in_dim // 8], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, C, height).permute(0, 2, 3, 1) # (batch_size, C, height, width)
        key = self.key_conv(x).view(batch_size, -1, C, width).permute(0, 2, 1, 3) # (batch_size, C, width, height)
        value = self.value_conv(x).view(batch_size, -1, C, height, width) # (batch_size, C, height, width)

        relative_h = self.relative_h[:height]
        relative_w = self.relative_w[:width]

        query = query + relative_h
        key = key + relative_w

        attention = torch.matmul(query, key) # (batch_size, C, height, width)
        attention = self.softmax(attention) # (batch_size, C, height, width)

        out = torch.matmul(value, attention.unsqueeze(2)).sum(dim=-1) # (batch_size, C, height, width)
        out = out.view(batch_size, C, height, width) # (batch_size, C, height, width)

        return out
