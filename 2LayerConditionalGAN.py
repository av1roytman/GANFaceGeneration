import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load digits dataset
digits = datasets.load_digits()

# Reshape data and scale to [-1, 1]
number_to_generate = 6
data = torch.tensor(digits.images[digits.target == number_to_generate] / 8 - 1).float().unsqueeze(1)  # Scale and add channel dimension
targets = torch.tensor(digits.target[digits.target == number_to_generate])
dataset = TensorDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Print some of the images
num_images_to_print = 16
def plot_images(images):
    plt.figure(figsize=(8, 8))
    for i in range(num_images_to_print):
        plt.subplot(round(math.sqrt(num_images_to_print)), round(math.sqrt(num_images_to_print)), i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.axis('off')
    plt.show()

plot_images(data)

# Define Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (batch_size, 100, 1, 1)
            # ConvTranspose2d will increase the spatial dimensions from 1x1 to 4x4
            nn.ConvTranspose2d(100, 128, 4, stride=1),  # Output size: (batch_size, 128, 4, 4)
            nn.BatchNorm2d(128),  # Batch normalization does not change the size: (batch_size, 128, 4, 4)
            nn.ReLU(True),  # ReLU activation does not change the size: (batch_size, 128, 4, 4)
            
            # ConvTranspose2d will increase the spatial dimensions from 4x4 to 8x8
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),  # Output size: (batch_size, 1, 8, 8)
            nn.Tanh()  # Tanh activation does not change the size: (batch_size, 1, 8, 8)
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (batch_size, 1, 8, 8)
            # Conv2d will reduce the spatial dimensions from 8x8 to 4x4
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(1, 128, 3, stride=2, padding=1),  # Output size: (batch_size, 128, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),

            # Conv2d will reduce the spatial dimensions from 4x4 to 1x1
            nn.Conv2d(128, 64, 4, stride=2),  # Output size: (batch_size, 64, 1, 1)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),  # Output size: (batch_size, 64)
            nn.Linear(64, 1),  # Output size: (batch_size, 1)
            nn.Sigmoid() # Size doesn't change. Only normalizes to [0, 1]
        )

    def forward(self, input):
        return self.main(input)
    
# Model Initialization
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002)

# Training Loop
num_epochs = 500
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(dataloader, 0):
        netD.zero_grad()
        real_data = data
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float, device=device)

        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        noise = torch.randn(batch_size, 100, 1, 1, device=device)

        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(1)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}')
            
fixed_noise = torch.randn(64, 100, 1, 1, device=device)

# After training, use the generator to produce images from the fixed noise vectors
with torch.no_grad():
    fake_images = netG(fixed_noise).cpu().numpy()

# Plot fake images
plot_images(fake_images)