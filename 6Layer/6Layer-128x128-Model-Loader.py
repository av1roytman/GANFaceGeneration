import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from Basic.Generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
