import torch
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
            nn.Flatten(),
            MinibatchDiscrimination(1024 * 4 * 4, 128, 16),
            nn.Linear(1024 * 4 * 4 + 128, 1),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.matmul(self.T.view(self.in_features, -1)) # NxBxC
        matrices = matrices.view(-1, self.out_features, self.kernel_dims) # NxBxC

        M = matrices.unsqueeze(0)  # 1xNxBxC, M_i,b in the paper
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC, M_j,b in the paper
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        norm_e = torch.exp(-norm) # exp(-||M_i,b - M_j,b||_L1)
        o_b = (norm_e.sum(0) - 1)   # NxB, subtract 1 because a vector is fully similar to itself but is counted in the exp(-norm) term

        x = torch.cat([x, o_b], 1)
        return x
