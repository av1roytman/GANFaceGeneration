import torch.nn as nn
import torch
import TransformerBlock

class Discriminator(nn.Module):
    def __init__(self, img_size, embed_dim, num_heads, num_layers, num_channels):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        # Convert input image to patches and then to a sequence of linear embeddings
        self.patch_size = 1  # Consider each pixel as a patch for simplicity
        self.num_patches = (img_size // self.patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_to_embed = nn.Linear(num_channels, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)  # Output layer for real vs fake classification
        )

    def forward(self, x):
        # Convert image patches to embeddings
        x = self.patch_to_embed(x.flatten(2)).transpose(1, 2)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(self.num_patches + 1)]
        x = self.dropout(x)

        x = self.transformer_blocks(x)

        cls_token = self.to_cls_token(x[:, 0])
        x = self.mlp_head(cls_token)
        return x