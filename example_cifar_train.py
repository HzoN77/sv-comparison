# Alot of code from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from datasets.loader import load_cifar10
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize from transform.
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__=="__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainloader = load_cifar10(transform, train=True)
    testloader = load_cifar10(transform, train=False)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))