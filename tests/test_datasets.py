import unittest
import torchvision.transforms as transforms
import torch
from datasets.loader import *


class TestDataSetMnist(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.ToTensor()

    def test_mnist(self):
        mnist = load_mnist(transform=self.transform, train=True)  # MNIST train batch size of 32
        dataiter = iter(mnist)
        batch = dataiter.next()
        image = batch[0]
        self.assertTrue(image.shape == (32, 1, 28, 28))  # Assert batch=32, and img sizes 28x28x1
        self.assertGreaterEqual(image.min(), 0)  # Assert values are between 0 and 1.
        self.assertGreaterEqual(0, image.max()-1)

    def test_cifar10(self):
        cifar = load_cifar10(transform=self.transform, train=False)  # CIFAR test with batch 32
        dataiter = iter(cifar)
        batch = dataiter.next()
        image = batch[0]
        self.assertTrue(image.shape == (32, 3, 32, 32))  # Assert image size and batch size 32x32x3
        self.assertGreaterEqual(image.min(), 0)  # Assert values are between 0 and 1.
        self.assertGreaterEqual(0, image.max() - 1)

    def test_tiny_imagenet(self):
        tiny = load_tiny_imagenet(transform=self.transform, train=False)  # CIFAR test with batch 32
        dataiter = iter(tiny)
        batch = dataiter.next()
        image = batch[0]
        self.assertTrue(image.shape == (32, 3, 64, 64))  # Assert image size and batch size 64x64x3
        self.assertGreaterEqual(image.min(), 0)  # Assert values are between 0 and 1.
        self.assertGreaterEqual(0, image.max() - 1)

    def test_batch_size(self):
        mnist = load_mnist(transform=self.transform, train=True, batch_size=1)
        self.assertEqual(len(mnist), 60000)  # 60000 training images
        with self.assertRaises(ValueError):
            load_mnist(transform=self.transform, train=True, batch_size=0)  # Batch size = 0 error

    def test_shuffle(self):
        mnist1 = load_mnist(transform=self.transform, train=False, batch_size=100, shuffle=False)
        mnist2 = load_mnist(transform=self.transform, train=False, batch_size=100, shuffle=False)
        batch1 = iter(mnist1).next()
        batch2 = iter(mnist2).next()
        for img1, img2 in zip(batch1, batch2):  # Compare one batch, make sure they are all equal
            self.assertTrue(torch.all(img1==img2).cpu().numpy() == 1)  # Assert images are equal. Typecast needed due to torch


if __name__ == "__main__":
    unittest.main()