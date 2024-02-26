import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Distributed Data Parallel
    dist.init_process_group('nccl')
    device = torch.device(f'cuda:{torch.distributed.get_rank()}')

    # Reshape data and scale to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
    ])

    image_dir = '../img_align_celeba'

    # Create a dataset
    dataset = CelebADataset(image_dir=image_dir, transform=transform)

    # Batch Size Hyperparameter
    global_batch_size = 64

    # Create a data loader
    # num_gpus = torch.cuda.device_count()
    # print("Number of available GPUs:", num_gpus)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=global_batch_size, shuffle=False, num_workers=8)

    # Create a 3x3 grid for the images
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    for i in range(9):
        image = dataset[i]
        image = (image + 1) / 2.0 # Scale images to [0, 1] to visualize better
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Directly use numpy and transpose here
        axes[i].axis('off')  # Turn off axes for cleaner look

    base = '../produced_images/SAGAN'
    model_base = '../checkpoints/SAGAN'

    plt.savefig(os.path.join(base, 'celeba_sample_128.png'))
    plt.close(fig)

    # Model Initialization
    netG = Generator()
    netD = Discriminator()

    # netG = netG.to(device)
    # netD = netD.to(device)

    # Distributed Data Parallel
    netG = DDP(netG.to(device))
    netD = DDP(netD.to(device))

    # Hyperparameters
    num_epochs = 150
    lr = 0.0001
    beta1 = 0
    beta2 = 0.9

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr*4, betas=(beta1, beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    dataloader_length = len(dataloader)

    gen_loss = []
    dis_loss = []
    batch_count = []

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(dataloader, 0):
            sampler.set_epoch(epoch)

            # Transfer data tensor to GPU/CPU (device)
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)

            # Train Discriminator
            netD.zero_grad()
            output = netD(real_data).view(-1)
            errD_real = torch.mean(torch.relu(1.0 - output))
            errD_real.backward()
            
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(torch.relu(1.0 + output))
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(1)  # The generator wants the discriminator to think the fake images are real
            output = netD(fake).view(-1)
            errG = -torch.mean(output)
            errG.backward()
            optimizerG.step()

            # Save Losses for plotting later

            if i % 50 == 0 and torch.distributed.get_rank() == 0:
                gen_loss.append(errG.item())
                dis_loss.append(errD.item())
                batch_count.append(i + dataloader_length * epoch)
                print(f'[{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

            if epoch % 25 == 0 and i == 0 and torch.distributed.get_rank() == 0:
                fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
                generate_images(netG, base, fixed_noise, label=f'Epoch-{epoch}')
                generate_loss_graphs(gen_loss, dis_loss, batch_count, base)
                torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-SAGAN.pth'))
                torch.save(netD.state_dict(), os.path.join(model_base, 'Dis-6Layer-128x128-SAGAN.pth'))

    print("Training is complete!")

    # Save the trained model
    torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-SAGAN.pth'))

    fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
    generate_images(netG, base, fixed_noise, label='Final')
    generate_loss_graphs(gen_loss, dis_loss, batch_count, base)


class CelebADataset(Dataset):
    def __init__(self, image_dir, transform=None, num_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        if num_samples:
            self.image_files = self.image_files[:num_samples]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
            
        return image


# Define Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            utils.spectral_norm(nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024 x 4 x 4

            utils.spectral_norm(nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8

            utils.spectral_norm(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 16 x 16

            # SelfAttention(256), # Self-Attention Layer

            utils.spectral_norm(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 32 x 32

            SelfAttention(128), # Self-Attention Layer
            # state size. 128 x 32 x 32

            utils.spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 64 x 64

            # SelfAttention(64), # Self-Attention Layer

            utils.spectral_norm(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)),  # Output 3 channels for RGB
            nn.Tanh()
            # state size. 3 x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 128 x 128
            utils.spectral_norm(nn.Conv2d(3, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1), # Dropout Layer
            # state size. 64 x 64 x 64

            # SelfAttention(64), # Self-Attention Layer

            utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1), # Dropout Layer
            # state size. 128 x 32 x 32

            SelfAttention(128), # Self-Attention Layer
            # state size. 128 x 32 x 32

            utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1), # Dropout Layer
            # state size. 256 x 16 x 16

            # SelfAttention(256), # Self-Attention Layer

            utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1), # Dropout Layer
            # state size. 512 x 8 x 8

            utils.spectral_norm(nn.Conv2d(512, 1024, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1), # Dropout Layer
            # state size. 1024 x 4 x 4

            utils.spectral_norm(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)),
            # state size. 1 x 1 x 1
            # nn.Sigmoid() # No Sigmoid function (bounded activation function) when using Hinge Loss
            # state size. 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1) # Remove the extra dimension


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self. out_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1) # (batch_size, height * width, C)
        key = self.key_conv(x).view(batch_size, -1, height * width) # (batch_size, C, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width) # (batch_size, C, height * width)

        attention = torch.bmm(query, key) # (batch_size, height * width, height * width)
        attention = self.softmax(attention) # (batch_size, height * width, height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1)) # (batch_size, C, height * width)
        out = out.view(batch_size, C, height, width) # (batch_size, C, height, width)
        out = self.out_conv(out) # (batch_size, C, height, width)

        return self.gamma * out + x  # Skip connection
    

def generate_images(netG, base, fixed_noise, label=""):
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
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Directly use numpy and transpose here
        axes[i].axis('off')  # Turn off axes for cleaner look

    plt.savefig(os.path.join(base, f'6Layer-128x128-SAGAN-{label}.png'))
    plt.close(fig)

    # return netG to training mode
    netG.train()

def generate_loss_graphs(gen_loss, dis_loss, batch_count, base):
    # Graph the Loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(batch_count, gen_loss, label="Generator")
    plt.plot(batch_count, dis_loss, label="Discriminator")
    plt.xlabel("Batch Count")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(base, f'Loss_6Layer-128x128-SAGAN.png'))
    plt.close()


if __name__ == '__main__':
    main()