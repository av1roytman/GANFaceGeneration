import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock
from GridTransformerBlock import GridTransformerBlock
from PositionalEncoding import PositionalEncoding

class Discriminator(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout, patch_size=4):
        super(Discriminator, self).__init__()
        self.embed_dim = embed_dim

        self.patch_sizes = [patch_size, patch_size * 2, patch_size * 4]

        self.patch_embeds = nn.ModuleList([PatchEmbedding(3, embed_dim // 4, self.patch_sizes[0]),
                                            PatchEmbedding(3, embed_dim // 4, self.patch_sizes[1]),
                                            PatchEmbedding(3, embed_dim // 2, self.patch_sizes[2])])

        image_size = 128 // patch_size

        self.pos_encs = nn.ModuleList([PositionalEncoding(embed_dim // 4, image_size * image_size),
                                        PositionalEncoding(embed_dim // 4, image_size // 2 * image_size // 2),
                                        PositionalEncoding(embed_dim // 2, image_size // 4 * image_size // 4)])

        # Stage 1: Transformer blocks and average pooling
        self.blocks_stage1 = nn.Sequential(*[GridTransformerBlock(embed_dim // 4, ff_dim, 32, 32, dropout) for _ in range(3)])
        self.avg_pool_stage1 = nn.AvgPool2d(kernel_size=(4,1), stride=(4,1))

        # Stage 2: Transformer blocks and average pooling
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim // 2, ff_dim, 16, 16, dropout) for _ in range(3)])
        self.avg_pool_stage2 = nn.AvgPool2d(kernel_size=(4,1), stride=(4,1))

        # Stage 3: Transformer blocks
        self.blocks_stage3 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 8, 8, dropout) for _ in range(3)])

        # Final transformer block and classification head
        self.final_block = TransformerBlock(embed_dim, ff_dim, 8, 8, dropout, token=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        # Input shape x: (batch_size, 3, 128, 128)
        assert x.shape == (batch_size, 3, 128, 128)

        xs = [patch_embed(x) for patch_embed in self.patch_embeds] # List of tensors of shape (batch_size, num_patches, embed_dim)
        assert xs[0].shape == (batch_size, 32*32, self.embed_dim // 4)
        assert xs[1].shape == (batch_size, 16*16, self.embed_dim // 4)
        assert xs[2].shape == (batch_size, 8*8, self.embed_dim // 2)

        xs = [self.pos_encs[i](xs[i]) for i in range(3)] # Add positional embeddings
        assert xs[0].shape == (batch_size, 32*32, self.embed_dim // 4)
        assert xs[1].shape == (batch_size, 16*16, self.embed_dim // 4)
        assert xs[2].shape == (batch_size, 8*8, self.embed_dim // 2)

        out = xs[0] # Size: (batch_size, 32*32, embed_dim / 4)
        assert out.shape == (batch_size, 32*32, self.embed_dim // 4)

        # Stage 1
        out = self.blocks_stage1(out) # Size: (batch_size, 32*32, embed_dim / 4)
        assert out.shape == (batch_size, 32*32, self.embed_dim // 4)

        out = self.avg_pool_stage1(out) # Size: (batch_size, 16*16, embed_dim / 4)
        assert out.shape == (batch_size, 16*16, self.embed_dim // 4)

        out = torch.cat([out, xs[1]], dim=2) # Size: (batch_size, 16*16, embed_dim / 2)
        assert out.shape == (batch_size, 16*16, self.embed_dim // 2)

        # Stage 2
        out = self.blocks_stage2(out) # Size: (batch_size, 16*16, embed_dim / 2)
        assert out.shape == (batch_size, 16*16, self.embed_dim // 2)

        out = self.avg_pool_stage2(out) # Size: (batch_size, 8*8, embed_dim / 2)
        assert out.shape == (batch_size, 8*8, self.embed_dim // 2)

        out = torch.cat([out, xs[2]], dim=2) # Size: (batch_size, 8*8, embed_dim)
        assert out.shape == (batch_size, 8*8, self.embed_dim)

        # Stage 3
        out = self.blocks_stage3(out) # Size: (batch_size, 8*8, embed_dim)
        assert out.shape == (batch_size, 8*8, self.embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Size: (batch_size, 1, embed_dim)
        assert cls_tokens.shape == (batch_size, 1, self.embed_dim)

        out = torch.cat([cls_tokens, out], dim=1) # Size: (batch_size, 65, embed_dim)
        assert out.shape == (batch_size, 65, self.embed_dim)

        out = self.final_block(out) # Size: (batch_size, 65, embed_dim)
        assert out.shape == (batch_size, 65, self.embed_dim)

        out = self.cls_head(out[:, 0]) # Size: (batch_size, 1)
        assert out.shape == (batch_size, 1)

        return out # Size: (batch_size, 1)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, _, height, width = x.shape
        num_patches = (height // self.patch_size) * (width // self.patch_size)

        x = self.proj(x)  # (batch_size, embed_dim, height/patch_size, width/patch_size)
        assert x.shape == (batch_size, self.embed_dim, height // self.patch_size, width // self.patch_size)

        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, embed_dim)
        assert x.shape == (batch_size, num_patches, self.embed_dim)

        return x

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels, embed_dim, patch_size):
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)  # (batch_size, embed_dim, height/patch_size, width/patch_size)
#         x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, embed_dim)
#         return x