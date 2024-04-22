import torch.nn as nn
from TransformerBlock import TransformerBlock
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, ff_dim, num_heads, dropout):
        super(Generator, self).__init__()

        self.embed_dim = embed_dim
        self.noise_dim = noise_dim

        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * embed_dim),
            nn.BatchNorm1d(7 * 7 * embed_dim),
            nn.ReLU(True)
        )

        self.pos_emb = nn.Parameter(torch.randn(1, 7 * 7, embed_dim))
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(3)])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bicubic')  # First upsampling from 7x7 to 14x14

        self.pos_emb2 = nn.Parameter(torch.randn(1, 14 * 14, embed_dim))
        self.blocks2 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(3)])
        self.upsample2 = nn.PixelShuffle(2)  # Second upsampling from 14x14 to 28x28

        self.pos_emb3 = nn.Parameter(torch.randn(1, 28 * 28, embed_dim // 4))
        self.blocks3 = nn.Sequential(*[TransformerBlock(embed_dim // 4, ff_dim, num_heads, dropout) for _ in range(3)])

        self.to_gray = nn.Conv2d(embed_dim // 4, 1, kernel_size=1)

        self.tanh = nn.Tanh()

    def forward(self, z):
        z = z.view(z.shape[0], -1) # size: (batch_size, noise_dim)
        x = self.mlp(z) # size: (batch_size, 7 * 7 * embed_dim)
        x = x.view(z.shape[0], 7 * 7, self.embed_dim) # size: (batch_size, 7 * 7, embed_dim)
        x = x + self.pos_emb # size: (batch_size, 7 * 7, embed_dim)
        x = self.blocks(x) # size: (batch_size, 7 * 7, embed_dim)
        x = x.transpose(1, 2) # size: (batch_size, embed_dim, 7 * 7)
        x = x.view(x.shape[0], self.embed_dim, 7, 7)
        x = self.upsample1(x)  # Upsample to 14x14
        # size: (batch_size, embed_dim, 14, 14)

        x = x.permute(0, 2, 3, 1)  # size: (batch_size, 14, 14, embed_dim)
        x = x.view(x.shape[0], 14 * 14, self.embed_dim) # size: (batch_size, 14 * 14, embed_dim)
        x = x + self.pos_emb2 # size: (batch_size, 14 * 14, embed_dim)
        x = self.blocks2(x) # size: (batch_size, 14 * 14, embed_dim)
        x = x.transpose(1, 2) # size: (batch_size, embed_dim, 14 * 14)
        x = x.view(x.shape[0], self.embed_dim, 14, 14) # size: (batch_size, embed_dim, 14, 14)
        x = self.upsample2(x)  # Upsample to 28x28
        # size: (batch_size, embed_dim // 4, 28, 28)

        x = x.permute(0, 2, 3, 1)  # size: (batch_size, 28, 28, embed_dim // 4)
        x = x.view(x.shape[0], 28 * 28, self.embed_dim // 4) # size: (batch_size, 28 * 28, embed_dim // 4)
        x = x + self.pos_emb3 # size: (batch_size, 28 * 28, embed_dim // 4)
        x = self.blocks3(x) # size: (batch_size, 28 * 28, embed_dim // 4)
        x = x.transpose(1, 2) # size: (batch_size, embed_dim // 4, 28 * 28)
        x = x.view(x.shape[0], self.embed_dim // 4, 28, 28) 
        # size: (batch_size, embed_dim, 28, 28)

        x = self.to_gray(x)  # Convert to grayscale size: (batch_size, 1, 28, 28)

        x = self.tanh(x)  # Normalize to [-1, 1]

        return x


class Discriminator(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout):
        super(Discriminator, self).__init__()

        self.embed_dim = embed_dim

        self.from_gray = PatchEmbedding(1, embed_dim, 4)
        
        self.pos_emb = nn.Parameter(torch.randn(1, (28 // 4) ** 2 + 1, embed_dim))

        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, num_heads, dropout) for _ in range(4)])

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.cls_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.from_gray(x)  # Convert to grayscale size: (batch_size, embed_dim, 28, 28)
        x = x.view(x.shape[0], (28 // 4) ** 2, self.embed_dim) # size: (batch_size, 7 * 7, embed_dim)

        # Concat CLS Token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_emb  # size: (batch_size, num_patches + 1, embed_dim)

        x = self.blocks(x)  # size: (batch_size, num_patches + 1, embed_dim)

        cls_representation = x[:, 0]  # CLS Token

        x = self.cls_head(cls_representation) # size: (batch_size, 1)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # size: (batch_size, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # size: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # size: (batch_size, num_patches, embed_dim)
        return x # size: (batch_size, num_patches, embed_dim)
    

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Reshape data and scale to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing to [-1, 1]
    ])

    # Create a dataset
    dataset = MNIST(root='./MNIST', train=True, download=True, transform=transform)

    # Batch Size Hyperparameter
    global_batch_size = 32

    dataloader = DataLoader(dataset, batch_size=global_batch_size, shuffle=True, num_workers=8)

    base = '../produced_images/TransGAN/MNIST'
    model_base = '../checkpoints/TransGAN/MNIST'

    version = '2'

    # Model Initialization
    netG = Generator(noise_dim=100, embed_dim=128, ff_dim=128, num_heads=8, dropout=0.1)
    netD = Discriminator(embed_dim=128, ff_dim=128, num_heads=8, dropout=0.2)

    netG = netG.to(device)
    netD = netD.to(device)

    # Hyperparameters
    num_epochs = 100
    lr = 0.001
    beta1 = 0
    beta2 = 0.99

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    dataloader_length = len(dataloader)

    gen_loss = []
    dis_loss = []
    batch_count = []

    netG.train()
    netD.train()

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        for i, (images, labels) in enumerate(dataloader, 0):
            real_data = images.to(device)
            batch_size = real_data.size(0)

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise)

            # Train Discriminator
            netD.zero_grad()
            real_output = netD(real_data).view(-1)
            errD_real = torch.mean(torch.relu(1.0 - real_output)).to(device)

            errD_real.backward()

            fake_output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(torch.relu(1.0 + fake_output)).to(device)

            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = -torch.mean(output).to(device)

            errG.backward()
            optimizerG.step()

            # Save Losses for plotting later

            if i % 50 == 0:
                gen_loss.append(errG.item())
                dis_loss.append(errD.item())
                batch_count.append(i + dataloader_length * epoch)
                print(f'[{version}][{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

                # # Print images and their labels from the discriminator
                # real_data = real_data.cpu().numpy().transpose((0, 2, 3, 1))
                # fake = fake.detach().cpu().numpy().transpose((0, 2, 3, 1))

                # fig, axs = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))

                # for j in range(batch_size):
                #     axs[0, j].imshow((real_data[j] * 0.5) + 0.5, cmap='gray')
                #     axs[0, j].set_title(f'Real: {real_output[j].item():.2f}')
                #     axs[0, j].axis('off')

                #     axs[1, j].imshow((fake[j] * 0.5) + 0.5, cmap='gray')
                #     axs[1, j].set_title(f'Fake: {fake_output[j].item():.2f}')
                #     axs[1, j].axis('off')

                # # print("I'm here")
                # plt.savefig(os.path.join(base, f'Samples/epoch_{epoch}_batch_{i}.png'))
                # plt.close(fig)

        if epoch % 5 == 0:
            fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
            generate_images(netG, base, fixed_noise, label1=f'{version}', label2=f'Epoch-{epoch}')
            torch.save(netG.state_dict(), os.path.join(model_base, f'Gen-TransGAN-MNIST-{version}-{epoch}.pth'))
            torch.save(netD.state_dict(), os.path.join(model_base, f'Dis-TransGAN-MNIST-{epoch}.pth'))

    print("Training is complete!")

    # Save the trained model

    torch.save(netG.state_dict(), os.path.join(model_base, f'Gen-TransGAN-MNIST-{version}.pth'))

    fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
    generate_images(netG, base, fixed_noise, label1=f'MNIST-{version}-Final')
    generate_loss_graphs(gen_loss, dis_loss, batch_count, base, label1=f'TransGAN-MNIST-{version}-final')


def generate_images(netG, base, fixed_noise, label1="", label2=""):
    # After training, use the generator to produce images from the fixed noise vectors
    netG.eval()
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    for i in range(9):
        image = fake_images[i]
        image = (image + 1) / 2.0 # Scale images to [0, 1] to visualize better
        image = image.squeeze()
        axes[i].imshow(image.numpy(), cmap='gray')  # Directly use numpy and transpose here
        axes[i].axis('off')  # Turn off axes for cleaner look

    plt.savefig(os.path.join(base, f'TransGAN-MNIST-{label1}-{label2}.png'))
    plt.close(fig)

    # return netG to training mode
    netG.train()

def generate_loss_graphs(gen_loss, dis_loss, batch_count, base, label1=""):
    # Graph the Loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(batch_count, gen_loss, label="Generator")
    plt.plot(batch_count, dis_loss, label="Discriminator")
    plt.xlabel("Batch Count")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(base, f'Loss_6Layer-128x128-{label1}.png'))
    plt.close()

if __name__ == "__main__":
    main()
