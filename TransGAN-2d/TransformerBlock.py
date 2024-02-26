import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, size, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.self_attn = SelfAttention(embed_dim, size, size)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Conv2d(embed_dim, ff_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(ff_dim, embed_dim, kernel_size=1),
        )

        self.ln1 = LayerNorm4D(embed_dim)
        self.ln2 = LayerNorm4D(embed_dim)

    def forward(self, x):
        # Input shape x: (batch_size, embed_dim, height, width)
        batch_size, embed_dim, height, width = x.shape

        attn_output = self.self_attn(x, height, width) # (batch_size, embed_dim, height, width)
        x = x + self.ln1(attn_output) # (batch_size, embed_dim, height, width)

        ffn_output = self.ffn(x) # (batch_size, embed_dim, height, width)
        x = x + self.ln2(ffn_output) # (batch_size, embed_dim, height, width)

        return x


class LayerNorm4D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size, channel, height, width = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x.view(-1, channel))
        x = x.view(batch_size, height, width, channel).permute(0, 3, 1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_dim, height, width):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self. out_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.positional_embeddings = nn.Parameter(torch.randn(1, in_dim, height, width))

    def forward(self, x):
        batch_size, C, height, width = x.size()

        x = x + self.positional_embeddings # (batch_size, C, height, width)

        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1) # (batch_size, height * width, C)
        key = self.key_conv(x).view(batch_size, -1, height * width) # (batch_size, C, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width) # (batch_size, C, height * width)

        attention = torch.bmm(query, key) # (batch_size, height * width, height * width)
        attention = self.softmax(attention) # (batch_size, height * width, height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1)) # (batch_size, C, height * width)
        out = out.view(batch_size, C, height, width) # (batch_size, C, height, width)
        out = self.out_conv(out) # (batch_size, C, height, width)

        return self.gamma * out + x  # Skip connection
