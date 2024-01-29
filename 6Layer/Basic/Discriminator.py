import torch.nn as nn

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), # Dropout Layer
            # state size. 64 x 64 x 64

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), # Dropout Layer
            # state size. 128 x 32 x 32

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), # Dropout Layer
            # state size. 256 x 16 x 16

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), # Dropout Layer
            # state size. 512 x 8 x 8

            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), # Dropout Layer
            # state size. 1024 x 4 x 4

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1) # Remove the extra dimension