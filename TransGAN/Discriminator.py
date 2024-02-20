import torch.nn as nn
import torch
from TransformerBlock import TransformerBlock
from GridTransformerBlock import GridTransformerBlock

class Discriminator(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
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
        self.final_block = TransformerBlock(embed_dim*4 + 1, ff_dim, dropout)
        self.cls_head = nn.Linear(embed_dim*4 + 1, 1)

    def forward(self, x):
        # Initial linear layer
        x = self.initial_linear(x) # Size: (batch_size, 32*32*embed_dim)
        x = x.view(-1, 32, 32, self.embed_dim) # Size: (batch_size, 32, 32, embed_dim)

        # Stage 1
        print("Embed_dim:", self.embed_dim)
        print("Stage 1 input shape:", x.shape)
        x = self.blocks_stage1(x) # Size: (batch_size, 32, 32, embed_dim)
        print("Stage 1 block 1 shape:", x.shape)
        x = self.avg_pool_stage1(x) # Size: (batch_size, 16, 16, embed_dim)
        print("Stage 1 avg pool shape:", x.shape)
        x = torch.cat([x, x], dim=1) # Size: (batch_size, 16, 16, embed_dim*2)
        print("Stage 1 output shape:", x.shape)

        # Stage 2
        print("Stage 2 input shape:", x.shape)
        x = self.blocks_stage2(x) # Size: (batch_size, 16, 16, embed_dim*2)
        x = self.avg_pool_stage2(x) # Size: (batch_size, 8, 8, embed_dim*2)
        x = torch.cat([x, x], dim=1) # Size: (batch_size, 8, 8, embed_dim*4)
        print("Stage 2 output shape:", x.shape)

        # Stage 3
        print("Stage 3 input shape:", x.shape)
        x = self.blocks_stage3(x) # Size: (batch_size, 8, 8, embed_dim*4)
        print("Stage 3 output shape:", x.shape)

        # Add CLS token
        cls_token = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) # Size: (batch_size, 8, 8, embed_dim*4)
        print("CLS token shape:", cls_token.shape)
        x = torch.cat([cls_token, x], dim=1)
        print("After cls concatenation:", x.shape)

        x = self.final_block(x) # Size: (batch_size, 1, 1, embed_dim*4)
        print("After final block:", x.shape)
        x = x.mean([2, 3])  # Size: (batch_size, embed_dim*4 + 1)
        print("After mean:", x.shape)
        x = self.cls_head(x) # Size: (batch_size, 1)
        print("After classification head:", x.shape)

        return x # Size: (batch_size, 1)
