import os
import time

import torch
import torchvision


def train_gan(device, generator, discriminator, dataloader, criterion, optimizer_g, optimizer_d,
              epochs=5, z_dim=100, save_interval=5, save_dir='./results'):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    real_label = 1
    fake_label = 0

    print("Starting Training Loop...")
    for epoch in range(epochs):

        start_time = time.time()

        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizer_d.zero_grad()

            # Train with real batch
            real_data = data.to(device)
            batch_size = real_data.size(0)
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_data).view(-1)
            errD_real = criterion(output, labels)
            errD_real.backward()
            output.mean().item()

            # Train with fake batch
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = generator(noise)
            labels.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            output.mean().item()

            errD = errD_real + errD_fake
            optimizer_d.step()

            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ###########################
            optimizer_g.zero_grad()
            labels.fill_(real_label)
            output = discriminator(fake).view(-1)
            errG = criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            # Update lists and print status
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        avg_G_loss = sum(G_losses[epoch * len(dataloader):]) / len(dataloader)
        avg_D_loss = sum(D_losses[epoch * len(dataloader):]) / len(dataloader)

        print(f"[{epoch}/{epochs}]\tAvg Discriminator Loss: {avg_D_loss:.2f}\tAvg Generator Loss: {avg_G_loss:.2f}\tin {time.time() - start_time:.2f}s")

        if epoch % save_interval == 0:
            # Save the generator's output
            with torch.no_grad():
                fake = generator(torch.randn(64, z_dim, 1, 1, device=device)).detach().cpu()
            img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))
            img_file = os.path.join(save_dir, 'epoch_{}.png'.format(epoch))
            torchvision.utils.save_image(fake, img_file)

            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(save_dir, 'generator_epoch_{}.pth'.format(epoch)))
            torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator_epoch_{}.pth'.format(epoch)))

    return G_losses, D_losses, img_list
