import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Reshape data and scale to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ColorJitter(),           # Data augmentation
    transforms.ToTensor(),
])

image_dir = 'img_align_celeba'

# Create a dataset
dataset = CelebADataset(image_dir=image_dir, transform=transform)

# Batch Size Hyperparameter
global_batch_size = 64

# Create a data loader
# num_gpus = torch.cuda.device_count()
# print("Number of available GPUs:", num_gpus)
dataloader = DataLoader(dataset, batch_size=global_batch_size, shuffle=True, pin_memory=True, num_workers=8)

# Create a 3x3 grid for the images
fig, axes = plt.subplots(3, 3, figsize=(9, 9))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

for i in range(9):
    image = dataset[i]
    axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Directly use numpy and transpose here
    axes[i].axis('off')  # Turn off axes for cleaner look

base = 'produced_images/5-layer'
plt.savefig(os.path.join(base, 'celeba_sample.png'))
plt.close(fig)

# Define Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # Output 3 channels for RGB
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            # state size. 1 x 1 x 1
            # nn.Sigmoid() #Commented because of LogitLoss
            # state size. 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1) # Remove the extra dimension

    
# Model Initialization
netG = Generator().to(device)
netD = Discriminator().to(device)

# Hyperparameters
num_epochs = 5
lr = 0.0002
beta1 = 0.5

# Binary cross entropy loss and optimizer
scaler = GradScaler()
criterion = BCEWithLogitsLoss()

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

dataloader_length = len(dataloader)

for epoch in range(1, num_epochs + 1):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device) * 2 - 1  # Scale images to [-1, 1]
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float, device=device)

        # Train Discriminator
        netD.zero_grad()
        # Wrap the forward pass in an autocast context
        with torch.cuda.amp.autocast():
            output = netD(real_data).view(-1)
            # Pass the pre-sigmoid output to criterion
            errD_real = criterion(output, label)
        # Scale the loss and call backward
        scaler.scale(errD_real).backward()
        
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0)
        with torch.cuda.amp.autocast():
            output = netD(fake.detach()).view(-1)
            # Pass the pre-sigmoid output to criterion
            errD_fake = criterion(output, label)
        # Scale the loss and call backward
        scaler.scale(errD_fake).backward()
        errD = errD_real + errD_fake
        # Step the optimizer and unscale the gradients
        scaler.step(optimizerD)
        scaler.update()

        # Train Generator
        netG.zero_grad()
        label.fill_(1)  # The generator wants the discriminator to think the fake images are real
        with torch.cuda.amp.autocast():
            output = netD(fake).view(-1)
            # Pass the pre-sigmoid output to criterion
            errG = criterion(output, label)
        # Scale the loss and call backward
        scaler.scale(errG).backward()
        # Step the optimizer and unscale the gradients
        scaler.step(optimizerG)
        scaler.update()

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

print("Training is complete!")


fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)

# After training, use the generator to produce images from the fixed noise vectors
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

plt.savefig(os.path.join(base, 'celeba_fake.png'))
plt.close(fig)