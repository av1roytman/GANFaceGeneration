import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load digits dataset instead of olivetti
digits = datasets.load_digits()
# Reshape data and scale to [-1, 1]
data = torch.tensor(digits.images / 8 - 1).float().unsqueeze(1)  # Scale and add channel dimension
targets = torch.tensor(digits.target)
dataset = TensorDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 3, stride=2),  # Adjusted size for 8x8 images
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)