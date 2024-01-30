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

base = 'produced_images/6-layer'
plt.savefig(os.path.join(base, 'celeba_sample_128.png'))
plt.close(fig)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 64 x 64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 32 x 32
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 16 x 16
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 8 x 8
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1024 x 4 x 4

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            # state size. 1 x 1 x 1
            # nn.Sigmoid() # Commented for Wasserstein
            # state size. 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1) # Remove the extra dimension
    
def compute_gradient_penalty(discriminator, real_images, fake_images, device):
    # Random weight term for interpolating between real and fake samples
    alpha = torch.rand((real_images.size(0), 1, 1, 1), device=device)
    alpha = alpha.expand_as(real_images)

    # Get random interpolation between real and fake samples
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    # Calculate discriminator output
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size(), device=device),
                                    create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Model Initialization
netG = Generator().to(device)
netD = Discriminator().to(device)

# Hyperparameters
num_epochs = 15
lr = 0.00002
beta1 = 0.5
lambda_gp = 10  # Clip value for discriminator weights

# Optimizers
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

dataloader_length = len(dataloader)

gen_loss = []
dis_loss = []
batch_count = []

# Training Loop
for epoch in range(1, num_epochs + 1):
    for i, data in enumerate(dataloader, 0):
        real_data = data.to(device)
        batch_size = real_data.size(0)

        # Train Discriminator
        for _ in range(5):  # Update the discriminator more frequently
            netD.zero_grad()
            real_loss = netD(real_data).mean()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise).detach()
            fake_loss = netD(fake).mean()

            errD = fake_loss - real_loss

            gradient_penalty = compute_gradient_penalty(netD, real_data, fake, device)
            errD += lambda_gp * gradient_penalty

            errD.backward()
            optimizerD.step()

        # Train Generator
        netG.zero_grad()
        fake = netG(noise)
        errG = -netD(fake).mean()
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
model_base = 'model_states/6-layer'
torch.save(netG.state_dict(), os.path.join(model_base, 'Gen-6Layer-128x128-Wasserstein.pth'))

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

plt.savefig(os.path.join(base, '6Layer-128x128-Wasserstein.png'))
plt.close(fig)

# Graph the Loss
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(batch_count, gen_loss, label="Generator")
plt.plot(batch_count, dis_loss, label="Discriminator")
plt.xlabel("Batch Count")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(base, 'Loss_6Layer-128x128-Wasserstein.png'))
plt.close()
