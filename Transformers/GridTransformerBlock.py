import torch.nn as nn
import torch

class GridSelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(GridSelfAttention, self).__init__()
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
        patch_size = 16

        # Split the input image into 16x16 patches
        x_patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x_patches = x_patches.contiguous().view(batch_size, C, -1, patch_size, patch_size)

        out_patches = []
        for i in range(x_patches.size(2)):
            # Process each patch separately
            x_patch = x_patches[:, :, i]

            query = self.query_conv(x_patch).view(batch_size, -1, patch_size * patch_size).permute(0, 2, 1)
            key = self.key_conv(x_patch).view(batch_size, -1, patch_size * patch_size)
            value = self.value_conv(x_patch).view(batch_size, -1, patch_size * patch_size)

            rel_pos_encoding = self.rel_pos_encoding(patch_size, patch_size).to(x.device)
            rel_pos_encoding = rel_pos_encoding.view(1, patch_size * patch_size, -1).permute(0, 2, 1)

            key = key + rel_pos_encoding

            attention = torch.bmm(query, key)
            attention = self.softmax(attention)

            out_patch = torch.bmm(value, attention.permute(0, 2, 1))
            out_patch = out_patch.view(batch_size, C, patch_size, patch_size)
            out_patch = self.out_conv(out_patch)

            out_patches.append(self.gamma * out_patch + x_patch)  # Skip connection

        # Concatenate the processed patches
        out = torch.cat(out_patches, dim=2).view(batch_size, C, height, width)

        return out

class GridTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(GridTransformerBlock, self).__init__()

        # Multi-Head Self-Attention
        self.self_attn = GridSelfAttention(embed_dim)

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
