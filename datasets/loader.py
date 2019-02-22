from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def load_mnist(transform=ToTensor(), train=True, batch_size=32, shuffle=True, num_workers=2):
    mnist = MNIST('./datasets/mnist/', train=train, transform=transform, download=True)
    return DataLoader(mnist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_cifar10(transform=ToTensor(), train=True, batch_size=32, shuffle=True, num_workers=2):
    mnist = CIFAR10('./datasets/cifar10/', train=train, transform=transform, download=True)
    return DataLoader(mnist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_tiny_imagenet(transform=ToTensor(), train=True, batch_size=32, shuffle=True, num_workers=2):
    from datasets.tinyimagenet import TinyImageNet
    tiny = TinyImageNet('./datasets/tiny-imagenet-200/val/images/', transform=transform)
    return DataLoader(tiny, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_omniglot(transform=ToTensor(), train=True, batch_size=32, shuffle=True, num_workers=2):
    pass

