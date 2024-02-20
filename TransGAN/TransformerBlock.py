import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Axial Attention
        self.self_attn = AxialAttention(embed_dim, max_len=128)

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
        print("TBlock Input shape:", x.shape)
        batch_size, embed_dim, height, width = x.shape

        # Apply Axial Attention
        attn_output = self.self_attn(x)

        # Reshape attn_output to match x's shape
        attn_output_reshaped = attn_output.contiguous().reshape(-1, embed_dim)

        # Reshape to 2D tensor before Layer Normalization
        x_reshaped = x.contiguous().reshape(-1, embed_dim)
        x_reshaped = self.ln1(x_reshaped + self.dropout(attn_output_reshaped))

        # Reshape back to 4D tensor after Layer Normalization
        x = x_reshaped.reshape(batch_size, embed_dim, height, width)

        # Apply FFN
        x = x.flatten(2).permute(2, 0, 1)  # Shape: (seq_length, batch_size, embed_dim)
        x = x.reshape(-1, embed_dim)  # Reshape x to have a last dimension of embed_dim
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))

        # Output shape: (batch_size, embed_dim, height, width)
        x = x.reshape(batch_size, embed_dim, height, width)
        return x


class AxialAttention(nn.Module):
    def __init__(self, in_dim, max_len=512):
        super(AxialAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.relative_h = nn.Parameter(torch.randn([in_dim // 8, max_len], dtype=torch.float32))
        self.relative_w = nn.Parameter(torch.randn([in_dim // 8, max_len], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()

        query = self.query_conv(x).view(batch_size, C // 8, height, width)  # (batch_size, C // 8, height, width)
        key = self.key_conv(x).view(batch_size, C // 8, height, width)  # (batch_size, C // 8, height, width)
        value = self.value_conv(x).view(batch_size, C, height, width)  # (batch_size, C, height, width)

        relative_h = self.relative_h[:, :height].unsqueeze(0).unsqueeze(-1)  # (1, C // 8, height, 1)
        relative_w = self.relative_w[:, :width].unsqueeze(0).unsqueeze(-2)  # (1, C // 8, 1, width)

        # print("Query shape:", query.shape)
        # print("Key shape:", key.shape)
        # print("Value shape:", value.shape)
        # print("Relative_h shape:", relative_h.shape)
        # print("Relative_w shape:", relative_w.shape)

        query = query + relative_h
        key = key + relative_w

        # print("Query shape after relative:", query.shape)
        # print("Key shape after relative:", key.shape)

        query = query.permute(0, 2, 3, 1)  # Transpose query for matrix multiplication
        key = key.permute(0, 2, 1, 3)  # Transpose key for matrix multiplication

        # print("Query shape after permute:", query.shape)
        # print("Key shape after permute:", key.shape)

        attention = torch.matmul(query, key)  # (batch_size, height, width, width)
        attention = self.softmax(attention)  # (batch_size, height, width, width)

        out = torch.matmul(attention, value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (batch_size, C, height, width)

        return out
