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