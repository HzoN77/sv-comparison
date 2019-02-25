# Alot of code from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from datasets.loader import load_cifar10
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.pytorch_eval import pytorch_train

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize from transform.
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    # Define a network
    net = Net()

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
    torch.save(net.state_dict(), os.path.join(modelpath, 'simpleNet-{}-epochs.pth'.format(10)))

    print('Finished Training')