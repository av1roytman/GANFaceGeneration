import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_images(netG, base, fixed_noise, label1="", label2=""):
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

    plt.savefig(os.path.join(base, f'6Layer-128x128-{label1}-{label2}.png'))
    plt.close(fig)

    # return netG to training mode
    netG.train()

def generate_loss_graphs(gen_loss, dis_loss, batch_count, base, label1=""):
    # Graph the Loss
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(batch_count, gen_loss, label="Generator")
    plt.plot(batch_count, dis_loss, label="Discriminator")
    plt.xlabel("Batch Count")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(base, f'Loss_6Layer-128x128-{label1}.png'))
    plt.close()