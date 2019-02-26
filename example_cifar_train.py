# Alot of code from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision.transforms as transforms
from datasets.loader import load_cifar10
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os

from models.cifar_example_net import Net
from utils.pytorch_eval import pytorch_train, pytorch_test

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

    # imshow(torchvision.utils.make_grid(images))

    # Define a network
    net = Net()

    # from models.vgg import VGG
    # net = VGG('VGG16')

    print("Network created")
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Create save directory
    modelpath = './models/saved/cifar10'
    if not os.path.isdir(modelpath):
        os.mkdir(modelpath)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Device set to", device)
    net.to(device)

    # Train the network
    for epoch in range(10):
        pytorch_train(model=net, dataloader=trainloader, device=device,
                      optimizer=optimizer, criterion=criterion, epoch=epoch)
        pytorch_test(model=net, device=device, dataloader=testloader)

    torch.save(net.state_dict(), os.path.join(modelpath, 'simpleNet-{}-epochs.pth'.format(10)))

    print('Finished Training')