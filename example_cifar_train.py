# Alot of code from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from datasets.loader import load_cifar10

if __name__=="__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainloader = load_cifar10(transform, train=True)
    testloader = load_cifar10(transform, train=False)
