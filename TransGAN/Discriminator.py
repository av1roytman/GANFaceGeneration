import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock
from GridTransformerBlock import GridTransformerBlock

class Discriminator(nn.Module):
    def __init__(self, embed_dim=96, ff_dim=192, dropout=0.1):
        super(Discriminator, self).__init__()
        self.embed_dim = embed_dim

        # Initial linear layer to map the input to the required dimensions
        self.initial_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3, 32*32*embed_dim),
            nn.ReLU(True)
        )

        # Stage 1: Transformer blocks and average pooling
        self.blocks_stage1 = nn.Sequential(*[GridTransformerBlock(embed_dim, ff_dim, 32, dropout) for _ in range(3)])
        self.avg_pool_stage1 = nn.AvgPool2d(2)

        # Stage 2: Transformer blocks and average pooling
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim*2, ff_dim, dropout) for _ in range(3)])
        self.avg_pool_stage2 = nn.AvgPool2d(2)

        # Stage 3: Transformer blocks
        self.blocks_stage3 = nn.Sequential(*[TransformerBlock(embed_dim*4, ff_dim, dropout) for _ in range(3)])

        # Final transformer block and classification head
        self.final_block = TransformerBlock(embed_dim*4, ff_dim, dropout)
        self.cls_head = nn.Linear(embed_dim*4, 1)

    def forward(self, x):
        # Initial linear layer
        x = self.initial_linear(x)
        x = x.view(-1, 32, 32, self.embed_dim)

        # Stage 1
        x = self.blocks_stage1(x)
        x = self.avg_pool_stage1(x)
        x = torch.cat([x, x], dim=1)

        # Stage 2
        x = self.blocks_stage2(x)
        x = self.avg_pool_stage2(x)
        x = torch.cat([x, x], dim=1)

        # Stage 3
        x = self.blocks_stage3(x)

        # Add CLS token
        cls_token = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
        x = torch.cat([cls_token, x], dim=1)

        # Final transformer block and classification head
        x = self.final_block(x)
        x = x[:, 0]
        x = self.cls_head(x)

        return x
