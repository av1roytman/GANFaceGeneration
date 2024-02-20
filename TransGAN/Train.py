import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from CelebADataset import CelebADataset
from Generator import Generator
from Discriminator import Discriminator
from Helpers import generate_images, generate_loss_graphs
from torch.cuda.amp import autocast, GradScaler

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
    global_batch_size = 2

    # Create a data loader
    # num_gpus = torch.cuda.device_count()
    # print("Number of available GPUs:", num_gpus)
    dataloader = DataLoader(dataset, batch_size=global_batch_size, shuffle=True, num_workers=8)

    # Create a 3x3 grid for the images
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    for i in range(9):
        image = dataset[i]
        image = (image + 1) / 2.0 # Scale images to [0, 1] to visualize better
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Directly use numpy and transpose here
        axes[i].axis('off')  # Turn off axes for cleaner look

    base = '../produced_images/TransGAN'
    model_base = '../model_states/TransGAN'

    plt.savefig(os.path.join(base, 'celeba_sample_128.png'))
    plt.close(fig)

    # Model Initialization
    # Model Initialization
    netG = Generator(noise_dim=100, embed_dim=64, ff_dim=2048, dropout=0.1)
    netD = Discriminator(embed_dim=64, ff_dim=2048, dropout=0.1)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     netG = nn.DataParallel(netG)
    #     netD = nn.DataParallel(netD)

    # Move the model to GPU
    netG.to(device)
    netD.to(device2)

    # Hyperparameters
    num_epochs = 150
    lr = 0.0001
    beta1 = 0
    beta2 = 0.99

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    dataloader_length = len(dataloader)

    gen_loss = []
    dis_loss = []
    batch_count = []

    scaler = GradScaler()

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(dataloader, 0):
            # Transfer data tensor to GPU/CPU (device)
            real_data = data.to(device2)
            batch_size = real_data.size(0)

            # Train Discriminator
            netD.zero_grad()
            with autocast():
                output = netD(real_data).view(-1)
                errD_real = torch.mean(torch.relu(1.0 - output))

            scaler.scale(errD_real).backward()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            with autocast():
                fake = netG(noise)
                output = netD(fake.detach()).view(-1)
                errD_fake = torch.mean(torch.relu(1.0 + output))

            scaler.scale(errD_fake).backward()
            errD = errD_real + errD_fake
            scaler.step(optimizerD)
            scaler.update()

            # Train Generator
            netG.zero_grad()
            with autocast():
                output = netD(fake).view(-1)
                errG = -torch.mean(output)

            scaler.scale(errG).backward()
            scaler.step(optimizerG)
            scaler.update()

            # Save Losses for plotting later

            if i % 50 == 0:
                gen_loss.append(errG.item())
                dis_loss.append(errD.item())
                batch_count.append(i + dataloader_length * epoch)
                print(f'[{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

            if epoch % 25 == 0 and i == 0:
                fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
                generate_images(netG, base, fixed_noise, label=f'Epoch-{epoch}')
                generate_loss_graphs(gen_loss, dis_loss, batch_count, base)
                torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-TransGAN.pth'))
                torch.save(netD.state_dict(), os.path.join(model_base, 'Dis-6Layer-128x128-TransGAN.pth'))

    print("Training is complete!")

    # Save the trained model
    torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-TransGAN.pth'))

    fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)
    generate_images(netG, base, fixed_noise, label='Final')
    generate_loss_graphs(gen_loss, dis_loss, batch_count, base)

if __name__ == '__main__':
    main()