import torch.nn as nn
import torch
import TransformerBlock

class Discriminator(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, ff_dim=2048, dropout=0.1):
        super(Discriminator, self).__init__()

        # Stage 1
        self.blocks_stage1 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(3)])
        self.average_pooling_stage1 = nn.AvgPool2d(kernel_size=, stride=, padding=)

        # Stage 2
        self.blocks_stage2 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(3)])
        self.average_pooling_stage2 = nn.AvgPool2d(kernel_size=, stride=, padding=)

        # Stage 3
        self.blocks_stage3 = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(3)])

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks_cls = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(1)])
        