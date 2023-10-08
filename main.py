# Main script for training GAN

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from dataset_loader import MemeDataset
from discriminator import Discriminator
from generator import Generator
from gan_training import train_gan  # Assuming train_gan_with_saving is in gan_training.py


if __name__ == '__main__':

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Device setup
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    # Hyperparameters
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    z_dim = 100
    batch_size = 64
    epochs = 500

    # Dataset and DataLoader setup
    dataset = MemeDataset(root_dir="memes", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, Loss function, and Optimizer setup
    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # Train the GAN
    G_losses, D_losses, img_list = train_gan(device, generator, discriminator, dataloader, criterion, optimizer_g,
                                             optimizer_d,
                                             epochs=epochs, save_interval=500, save_dir='./results')

    print("Training complete!")
