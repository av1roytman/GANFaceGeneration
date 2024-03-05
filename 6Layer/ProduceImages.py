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
from MoreTrainSAGAN import Generator

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    netG = Generator().to(device)
    netG.load_state_dict(torch.load('../model_states/SAGAN/Gen-6Layer-128x128-SAGAN-Big3.pth'))

    for i in range(4):
        fixed_noise = torch.randn(4, 100, 1, 1, device=device)
        generate_images(netG, '../produced_images/SAGAN', fixed_noise, 4, label=f"{i}")

def generate_images(netG, base, fixed_noise, num_images, label=""):
    # After training, use the generator to produce images from the fixed noise vectors
    netG.eval()
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()

    for i in range(num_images):
        image = fake_images[i]
        image = (image + 1) / 2.0 # Scale images to [0, 1] to visualize better
        axes[i].imshow(np.transpose(image.numpy(), (1, 2, 0)))  # Directly use numpy and transpose here
        axes[i].axis('off')  # Turn off axes for cleaner look

    plt.savefig(os.path.join(base, f'SAGAN-Produced-{label}.png'))
    plt.close(fig)


if __name__ == "__main__":
    main()