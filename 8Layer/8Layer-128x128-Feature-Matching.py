import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os
from Basic.Generator import Generator

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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
])

image_dir = 'img_align_celeba'

# Create a dataset
dataset = CelebADataset(image_dir=image_dir, transform=transform)

# Batch Size Hyperparameter
global_batch_size = 64

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

base = 'produced_images/8-layer'
plt.savefig(os.path.join(base, 'celeba_sample_128.png'))
plt.close(fig)


# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 64 x 64 x 64

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 128 x 32 x 32

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 256 x 16 x 16

            # Adding two more layers
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 256 x 16 x 16
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 512 x 16 x 16

            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 512 x 8 x 8

            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),  # Dropout Layer
            # state size. 1024 x 4 x 4
            nn.Conv2d(1024, 1, 4, stride=1, padding=0),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input, feature_matching=False):
        if feature_matching:
            # Extract features from an intermediate layer
            features = self.main[:21](input) 
            return features
        else:
            return self.main(input).view(-1, 1).squeeze(1)

    
def add_noise_to_image(image, mean=0.0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    return image + noise

# Model Initialization
netG = Generator().to(device)
netD = Discriminator().to(device)

# Hyperparameters
num_epochs = 5
lr = 0.00002
beta1 = 0.5

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

dataloader_length = len(dataloader)

gen_loss = []
dis_loss = []
batch_count = []

feature_matching_loss_weight = 0.5
adversarial_loss_weight = 1

# Training Loop
for epoch in range(1, num_epochs + 1):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        real_data = add_noise_to_image(real_data)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float, device=device)

        # Train Discriminator
        netD.zero_grad()
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = netG(noise)
        fake = add_noise_to_image(fake)
        label.fill_(0)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label.fill_(1)  # The generator wants the discriminator to think the fake images are real

        # Forward pass through discriminator with feature matching
        fake = netG(noise)
        fake = add_noise_to_image(fake)

        # Extract Features
        features_fake = netD(fake, feature_matching=True)
        features_real = netD(real_data, feature_matching=True)

        # Calculate feature matching loss
        feature_matching_loss = F.mse_loss(features_fake, features_real.detach())

        # Calculate adversarial loss
        output = netD(fake).view(-1)
        adversarial_loss = criterion(output, label)

        # Combine losses (you may want to weight these losses)
        errG = adversarial_loss_weight * adversarial_loss + feature_matching_loss_weight * feature_matching_loss
        errG.backward()
        optimizerG.step()

        # Save Losses for plotting later
        gen_loss.append(errG.item())
        dis_loss.append(errD.item())
        batch_count.append(i + dataloader_length * epoch)

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

print("Training is complete!")

# Save the trained model
model_base = 'model_states/8-layer'
torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-8Layer-128x128.pth'))

fixed_noise = torch.randn(global_batch_size, 100, 1, 1, device=device)

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

plt.savefig(os.path.join(base, '8Layer-128x128.png'))
plt.close(fig)

# Graph the Loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(batch_count, gen_loss, label="Generator")
plt.plot(batch_count, dis_loss, label="Discriminator")
plt.xlabel("Batch Count")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(base, 'Loss_8Layer-128x128.png'))
plt.close()
