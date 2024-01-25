import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 64 x 64

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # Output 3 channels for RGB
            nn.Tanh()
            # state size. 3 x 128 x 128
        )

    def forward(self, input):
        return self.main(input)

def generate_images(model, num_images=4):
    # Assuming the model takes random noise as input and generates an image
    # Adjust the size of the noise vector according to your model's requirements
    noise = torch.randn(num_images, 100, 1, 1, device=device)
    netG = model.to(device)
    netG.eval()
    with torch.no_grad():
        images = netG(noise)
    return images

def main(model_path):
    # Load the model
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate images
    images = generate_images(model)

    # Create a grid of images
    grid = make_grid(images, nrow=2)

    grid = grid.cpu().numpy()

    grid = (grid - grid.min()) / (grid.max() - grid.min())

    # Display the grid
    plt.imshow(grid.transpose(1, 2, 0))
    plt.axis('off')
    plt.show()

    # Save the grid
    plt.imsave('produced_images/4x4ForPresentation.png', grid.transpose(1, 2, 0))

if __name__ == "__main__":
    model_path = 'model_states/6-layer/Gen-6Layer-128x128.pth'
    main(model_path)
