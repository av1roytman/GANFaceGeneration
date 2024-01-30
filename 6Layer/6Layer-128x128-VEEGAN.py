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
from Basic.Discriminator import Discriminator

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

base = 'produced_images/6-layer'
plt.savefig(os.path.join(base, 'celeba_sample_128.png'))
plt.close(fig)

# Define Reconstructor
class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 128 * 128, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 100)  # Matching the Generator's input size
        )

    def forward(self, input):
        return self.main(input).view(-1, 100, 1, 1)
    
def add_noise_to_image(image, mean=0.0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    return image + noise

# Model Initialization
netG = Generator().to(device)
netD = Discriminator().to(device)
netR = Reconstructor().to(device)

# Hyperparameters
num_epochs = 15
lr = 0.00002
beta1 = 0.5

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerR = torch.optim.Adam(netR.parameters(), lr=lr, betas=(beta1, 0.999))

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()

dataloader_length = len(dataloader)

gen_loss = []
dis_loss = []
rec_loss = []
batch_count = []

# Training Loop
for epoch in range(1, num_epochs + 1):
    for i, data in enumerate(dataloader, 0):
        # Transfer data tensor to GPU/CPU (device)
        real_data = data.to(device)
        real_data = add_noise_to_image(real_data)

        batch_size = real_data.size(0)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # Create the noise vector
        noise = torch.randn(batch_size, 100, 1, 1, device=device)

        # Generate fake image batch with G
        fake = netG(noise)
        fake = add_noise_to_image(fake)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizerD.zero_grad()

        # Compute Discriminator Loss on Real and Fake Data
        real_output = netD(real_data).view(-1)
        loss_D_real = bce_loss(real_output, real_labels)

        fake_output = netD(fake.detach()).view(-1)
        loss_D_fake = bce_loss(fake_output, fake_labels)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizerD.step()

        # Train Generator and Reconstructor
        optimizerG.zero_grad()
        fake_output = netD(fake).view(-1)
        loss_G = bce_loss(fake_output, real_labels)

        optimizerR.zero_grad()
        reconstructed_noise = netR(fake)
        loss_R = mse_loss(reconstructed_noise, noise)

        total_loss_G = loss_G + loss_R
        total_loss_G.backward()
        optimizerG.step()
        optimizerR.step()

        # Save Losses for plotting later
        gen_loss.append(loss_G.item())
        dis_loss.append(loss_D.item())
        rec_loss.append(loss_R.item())
        batch_count.append(i + dataloader_length * epoch)

        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{dataloader_length}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} Loss_R: {loss_R.item():.4f}')

print("Training is complete!")

# Save the trained model
model_base = 'model_states/6-layer'
torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-VEEGAN.pth'))

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

plt.savefig(os.path.join(base, '6Layer-128x128-VEEGAN.png'))
plt.close(fig)

# Graph the Loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(batch_count, gen_loss, label="Generator")
plt.plot(batch_count, dis_loss, label="Discriminator")
plt.plot(batch_count, rec_loss, label="Reconstructor")
plt.xlabel("Batch Count")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(base, 'Loss_6Layer-128x128-VEEGAN.png'))
plt.close()
