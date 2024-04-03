import torch.nn as nn
from TransformerBlock import TransformerBlock
from PositionalEncoding import PositionalEncoding
from Generator import UpsamplingBlock, UpsampleBlock_PixelShuffle
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from Helpers import generate_images, generate_loss_graphs
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, ff_dim, dropout):
        super(Generator, self).__init__()

        self.embed_dim = embed_dim
        self.noise_dim = noise_dim

        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * embed_dim),
            nn.ReLU(True)
        )

        self.pos_enc = PositionalEncoding(embed_dim, 7 * 7)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 7, 7, dropout) for _ in range(1)])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bicubic')  # First upsampling from 7x7 to 14x14

        self.pos_enc2 = PositionalEncoding(embed_dim, 14 * 14)
        self.blocks2 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 14, 14, dropout) for _ in range(1)])
        self.upsample2 = nn.PixelShuffle(2)  # Second upsampling from 14x14 to 28x28

        self.pos_enc3 = PositionalEncoding(embed_dim // 4, 28 * 28)
        self.blocks3 = nn.Sequential(*[TransformerBlock(embed_dim // 4, ff_dim, 28, 28, dropout) for _ in range(1)])

        self.to_gray = nn.Conv2d(embed_dim // 4, 1, kernel_size=1)  # Ensure this matches the channel size after upsampling

    def forward(self, z):
        z = z.view(z.shape[0], -1)
        x = self.mlp(z)
        x = x.view(z.shape[0], 7 * 7, self.embed_dim)
        x = self.pos_enc(x)
        x = self.blocks(x)
        x = x.view(x.shape[0], self.embed_dim, 7, 7)
        x = self.upsample1(x)  # Upsample to 14x14
        # size: (batch_size, embed_dim, 14, 14)

        x = x.view(x.shape[0], 14 * 14, self.embed_dim)
        x = self.pos_enc2(x)
        x = self.blocks2(x)
        x = x.view(x.shape[0], self.embed_dim, 14, 14)
        x = self.upsample2(x)  # Upsample to 28x28
        # size: (batch_size, embed_dim, 28, 28)

        x = x.view(x.shape[0], 28 * 28, self.embed_dim // 4)
        x = self.pos_enc3(x)
        x = self.blocks3(x)
        x = x.view(x.shape[0], self.embed_dim // 4, 28, 28)
        # size: (batch_size, embed_dim, 28, 28)

        x = self.to_gray(x)  # Convert to grayscale
        return x


class Discriminator(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super(Discriminator, self).__init__()

        self.embed_dim = embed_dim

        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 28, 28, dropout) for _ in range(1)])
        self.from_gray = nn.Conv2d(1, embed_dim, kernel_size=1)  # Ensure this matches the channel size after upsampling

        self.pos_enc2 = PositionalEncoding(embed_dim, 28 * 28)
        self.blocks2 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 28, 28, dropout) for _ in range(1)])
        self.downsample1 = nn.AvgPool2d(kernel_size=2)  # First downsampling from 28x28 to 14x14

        self.pos_enc3 = PositionalEncoding(embed_dim, 14 * 14)
        self.blocks3 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 14, 14, dropout) for _ in range(1)]
                                    + [nn.LayerNorm(embed_dim)])

        self.downsample2 = nn.AvgPool2d(kernel_size=2)  # Second downsampling from 14x14 to 7x7

        self.pos_enc4 = PositionalEncoding(embed_dim, 7 * 7)
        self.blocks4 = nn.Sequential(*[TransformerBlock(embed_dim, ff_dim, 7, 7, dropout) for _ in range(1)]
                                    + [nn.LayerNorm(embed_dim)])

        self.to_prob = nn.Linear(7 * 7 * embed_dim, 1)

    def forward(self, x):
        x = self.from_gray(x)  # Convert to grayscale
        x = x.view(x.shape[0], 28 * 28, self.embed_dim)
        x = self.blocks(x)
        x = x.view(x.shape[0], 28 * 28, self.embed_dim)
        x = self.pos_enc2(x)
        x = self.blocks2(x)
        x = x.view(x.shape[0], self.embed_dim, 28, 28)
        x = self.downsample1(x)  # Downsample to 14x14
        # size: (batch_size, embed_dim, 14, 14)

        x = x.view(x.shape[0], 14 * 14, self.embed_dim)
        x = self.pos_enc3(x)
        x = self.blocks3(x)
        x = x.view(x.shape[0], self.embed_dim, 14, 14)
        x = self.downsample2(x)
        # size: (batch_size, embed_dim, 7, 7)

        x = x.view(x.shape[0], 7 * 7, self.embed_dim)
        x = self.pos_enc4(x)
        x = self.blocks4(x)
        x = x.view(x.shape[0], -1)
        x = self.to_prob(x)

        return x
    

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    # Model Initialization
    netG = Generator(noise_dim=100, embed_dim=256, ff_dim=64, dropout=0.1)
    netD = Discriminator(embed_dim=256, ff_dim=64, dropout=0.1)

    netG = netG.to(device)
    netD = netD.to(device)

    # Hyperparameters
    num_epochs = 10
    lr = 0.001
    beta1 = 0.5
    beta2 = 0.99

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    dataloader_length = len(dataloader)

    gen_loss = []
    dis_loss = []
    batch_count = []


    # Training Loop
    for epoch in range(1, num_epochs + 1):
        for i, (images, labels) in enumerate(dataloader, 0):
            # Transfer data tensor to GPU/CPU (device)
            real_data = images.to(device)
            batch_size = real_data.size(0)

            # Train Discriminator
            netD.zero_grad()
            real_output = netD(real_data).view(-1)
            errD_real = torch.mean(torch.relu(1.0 - real_output))

            errD_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise)
            fake_output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(torch.relu(1.0 + fake_output))

            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            output = netD(fake).view(-1)
            errG = -torch.mean(output)

            errG.backward()
            optimizerG.step()

            # Save Losses for plotting later

            if i % 50 == 0:
                gen_loss.append(errG.item())
                dis_loss.append(errD.item())
                batch_count.append(i + dataloader_length * epoch)
                print(f'[{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

                # Print images and their labels from the discriminator
                real_data = real_data.cpu().numpy().transpose((0, 2, 3, 1))
                fake = fake.detach().cpu().numpy().transpose((0, 2, 3, 1))

                fig, axs = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))

                for j in range(batch_size):
                    axs[0, j].imshow((real_data[j] * 0.5) + 0.5, cmap='gray')
                    axs[0, j].set_title(f'Real: {real_output[j].item():.2f}')
                    axs[0, j].axis('off')

                    axs[1, j].imshow((fake[j] * 0.5) + 0.5, cmap='gray')
                    axs[1, j].set_title(f'Fake: {fake_output[j].item():.2f}')
                    axs[1, j].axis('off')

                # print("I'm here")
                plt.savefig(os.path.join(base, f'Samples/epoch_{epoch}_batch_{i}.png'))

            if epoch % 10 == 0 and i == 0:
                fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
                generate_images(netG, base, fixed_noise, label1='TransGAN-MNIST', label2=f'Epoch-{epoch}')
                torch.save(netG.state_dict(), os.path.join(model_base, f'Gen-6Layer-128x128-TransGAN-MNIST-{epoch}.pth'))
                torch.save(netD.state_dict(), os.path.join(model_base, f'Dis-6Layer-128x128-TransGAN-MNIST-{epoch}.pth'))

    print("Training is complete!")

    # Save the trained model
    torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-TransGAN-MNIST.pth'))

    fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
    generate_images(netG, base, fixed_noise, label='Final')
    generate_loss_graphs(gen_loss, dis_loss, batch_count, base)


if __name__ == "__main__":
    main()
