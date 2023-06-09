import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self.block(features_d, features_d*2, 4, 2, 1),
            self.block(features_d*2, features_d * 4, 4, 2, 1),
            self.block(features_d*4, features_d * 8, 4, 2, 1),

            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.block(z_dim, features_g*16, 4, 1, 0),
            self.block(features_g*16, features_g*8,  4, 2, 1),
            self.block(features_g* 8, features_g*4, 4, 2, 1),
            self.block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(),
        )
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(0.2)


        )
    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(disc)
    initialize_weights(gen)
    assert disc(x).shape == (N, 1, 1, 1)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)

def saveimage(tensorlist, filepath, unique:bool=False):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if unique:
        x = 1
        for file in os.listdir(filepath):
            if os.path.isfile(os.path.join(os.getcwd(), filepath, file)):
                x += 1

    else:
        x = 1
    firstdir = os.getcwd()
    dr = os.path.join(os.getcwd(), filepath)
    for a in tensorlist:
        with torch.no_grad():
            numpy_image = a.cpu().permute(1, 2, 0).numpy()

            numpy_image = numpy_image * 255
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
            img = cv2.imshow("image", numpy_image)
            os.chdir(dr)
            cv2.imwrite("img"+str(x)+".png", numpy_image)

        x += 1
    os.chdir(firstdir)
#test()

def showimage(tensorv):
    with torch.no_grad():
        numpy_image = tensorv.cpu().permute(1, 2, 0).numpy()

        numpy_image = numpy_image * 255
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("image", numpy_image)
        cv2.waitKey()

    """
    transform = transforms.ToPILImage()
    img = transform(tensor)
    img.show()
    """

