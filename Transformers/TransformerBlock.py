import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self. out_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def rel_pos_encoding(self, height, width):
        # Compute relative position encoding
        col_indices = torch.arange(width).unsqueeze(0).repeat(height, 1)
        row_indices = torch.arange(height).unsqueeze(1).repeat(1, width)
        col_diff = col_indices.unsqueeze(0) - col_indices.unsqueeze(1)
        row_diff = row_indices.unsqueeze(0) - row_indices.unsqueeze(1)
        rel_pos_encoding = torch.stack((col_diff, row_diff), dim=-1).float()

        return rel_pos_encoding

    def forward(self, x):
        batch_size, C, height, width = x.size()

        query = self.query_conv(x).view(batch_size, C, -1).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, C, -1).permute(0, 2, 1)
        value = self.value_conv(x).view(batch_size, C, -1).permute(0, 2, 1)

        rel_pos_encoding = self.rel_pos_encoding(height, width).to(x.device)
        rel_pos_encoding = rel_pos_encoding.view(1, height * width, -1).permute(0, 2, 1)

        key = key + rel_pos_encoding

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-Head Self-Attention
        self.self_attn = SelfAttention(embed_dim)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
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
        batch_size, height, width, _ = x.shape

        # Apply 2D MHSA
        x = x.view(batch_size, height * width, -1).permute(1, 0, 2)  # Shape: (seq_length, batch_size, embed_dim)
        attn_output = self.self_attn(x)
        x = self.ln1(x + self.dropout(attn_output)).permute(1, 0, 2)  # Shape: (batch_size, seq_length, embed_dim)

        # Apply FFN
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))

        # Output shape: (batch_size, height, width, embed_dim)
        return x.view(batch_size, height, width, -1)