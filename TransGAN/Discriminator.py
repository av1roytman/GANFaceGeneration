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

        self.pos_embeds = nn.ParameterList([nn.Parameter(torch.randn(1, image_size**2, embed_dim // 4)),
                                            nn.Parameter(torch.randn(1, (image_size // 2)**2, embed_dim // 4)),
                                            nn.Parameter(torch.randn(1, (image_size // 4)**2, embed_dim // 2))])

        self.pos_encs = nn.ModuleList([PositionalEncoding(embed_dim // 4, image_size, image_size),
                                        PositionalEncoding(embed_dim // 4, image_size // 2, image_size // 2),
                                        PositionalEncoding(embed_dim // 2, image_size // 4, image_size // 4)])

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

        xs = [patch_embed(x) for patch_embed in self.patch_embeds] # List of tensors of shape (batch_size, num_patches, embed_dim)
        # print shapes
        # for i in range(3):
        #     print(f'xs[{i}]: {xs[i].shape}')

        xs = [xs[i] + self.pos_embeds[i] for i in range(3)] # Add positional embeddings

        out = xs[0] # Size: (batch_size, 32*32, embed_dim / 4)

        # Stage 1
        out = self.blocks_stage1(out) # Size: (batch_size, 32*32, embed_dim / 4)
        # print(f'out after block: {out.shape}')
        out = self.avg_pool_stage1(out) # Size: (batch_size, 16*16, embed_dim / 4)
        # print(f'out: {out.shape}')
        # print(f'xs[1]: {xs[1].shape}')
        out = torch.cat([out, xs[1]], dim=2) # Size: (batch_size, 16*16, embed_dim / 2)

        # Stage 2
        out = self.blocks_stage2(out) # Size: (batch_size, 16*16, embed_dim / 2)
        out = self.avg_pool_stage2(out) # Size: (batch_size, 8*8, embed_dim / 2)
        out = torch.cat([out, xs[2]], dim=2) # Size: (batch_size, 8*8, embed_dim)

        # Stage 3
        out = self.blocks_stage3(out)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat([cls_tokens, out], dim=1)

        out = self.final_block(out)

        out = self.cls_head(out)

        return out



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, height/patch_size, width/patch_size)
        # print(f'This x: {x.shape}')
        x = x.flatten(2).transpose(1, 2) # (batch_size, num_patches, embed_dim)
        # print(f'That x: {x.shape}')
        return x