import os
import tarfile
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.data, self.targets = self._load_data(train)

    def _load_data(self, train):
        if train:
            batch_files = [os.path.join(self.root, 'data_batch_{}'.format(i)) for i in range(1, 6)]
        else:
            batch_files = [os.path.join(self.root, 'test_batch')]

        data, targets = [], []
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])
        
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to HWC
        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, target

# Define the Vision Transformer model
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] => [B, E, H', W']
        x = x.flatten(2)  # [B, E, H', W'] => [B, E, N] where N=H'*W'
        x = x.transpose(1, 2)  # [B, E, N] => [B, N, E] to fit the Transformer's input format
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        return self.encoder(src)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=768, num_heads=8, num_layers=6, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
print("Loading CIFAR-10 dataset")
train_dataset = CIFAR10Dataset(root='../cifar-10', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = CIFAR10Dataset(root='../cifar-10', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, embed_dim=768, num_heads=8, num_layers=6, num_classes=10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

def evaluate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

print("Training")
train(model, train_loader, criterion, optimizer, num_epochs=10)

print("Evaluating")
evaluate(model, test_loader)
