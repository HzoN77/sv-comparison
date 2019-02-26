import torch.utils.data as data
from PIL import Image
import os

class TinyImageNet(data.Dataset):
    def __init__(self, root=None, download=True, transform=None):
        self.root = os.path.abspath(root)
        self.download = download
        self.transform = transform
        self.paths, self.labels = self.load_paths()
        self.length = len(self.paths)

    def load_paths(self):
        self.check_if_downloaded()
        paths = []
        for idx in range(10000):
            filename = 'val_' + str(idx) + '.JPEG'
            paths.append(os.path.join(self.root, filename))

        labels = [-1] * len(paths)
        return paths, labels

    def check_if_downloaded(self):
        if not os.path.isdir(self.root):
            print(self.root)
            if not self.download:
                raise AssertionError("Could not find dataset")
            else:
                download_tiny_imagenet_validation()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        img = Image.open(self.paths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def download_tiny_imagenet_validation():
    import zipfile
    import os
    import sys
    import urllib.request
    print("Downloading Tiny ImageNet")
    download = urllib.request.urlretrieve('http://cs231n.stanford.edu/tiny-imagenet-200.zip', filename='./datasets/tiny-imagenet-200.zip')
    archive = zipfile.ZipFile(download[0])  # Take the path of the downloaded file

    print("Extracting validation data ...")
    sys.stdout = open(os.devnull, 'w')
    for file in archive.namelist():
        if 'val/' in file:
            archive.extract(file, './datasets/')
    sys.stdout = sys.__stdout__