import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8

            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    checkpoint = torch.load("checkpoint.pth")
    manualseed = random.randint(0, 10000)
    print("Random Seed: ", manualseed)
    random.seed(manualseed)
    torch.manual_seed(manualseed)

    dataroot = "venv/waifu"
    workers = 2
    batch_size = 8
    image_size = 128
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    epochs = 200
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    real_batch = next(iter(dataloader))

    netG = Generator(ngpu).to(device)
    if device.type == "cuda" and ngpu > 1:
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.
    netG.load_state_dict(checkpoint["netG"])
    netD.load_state_dict(checkpoint["netD"])
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD.load_state_dict(checkpoint["optimizerD"])
    optimizerG.load_state_dict(checkpoint["optimizerG"])
    img_list = checkpoint["img_list"]
    G_losses = checkpoint["g_losses"]
    D_losses = checkpoint["d_losses"]
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(epochs):

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)

            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)

            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)

            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)

            errG = criterion(output, label)

            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    real_batch = next(iter(dataloader))

    # Plot the real images

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=3000, repeat_delay=1000, blit=True)
    f = open("output.html", "w")

    template = f"""<html>
    <head>
    <title>Title</title>
    </head>
    <body>
    <h2>Welcome To GFG</h2>

    {ani.to_html5_video()}

    </body>
    </html>
    """
    f.write(template)
    f.close()
    data = {
        "optimizerD": optimizerD.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "netD": netD.state_dict(),
        "netG": netG.state_dict(),
        "img_list": img_list,
        "g_losses": G_losses,
        "d_losses": D_losses
    }
    torch.save(data, "checkpoint.pth")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
