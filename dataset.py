import os

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets


def load_mnist(isTrainSet=True, withJitter=False):
    transform_list = [torchvision.transforms.ToTensor()]
    if withJitter:
        transform_list.append(torchvision.transforms.GaussianBlur(3, 0.5))
    dset = datasets.MNIST(
        os.path.dirname(os.path.realpath(__file__)) + "/data",
        train=isTrainSet,
        download=True,
        transform=torchvision.transforms.Compose(transform_list),
    )
    return dset


def get_dataloader(dataset, batchSize=100):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchSize, shuffle=True, drop_last=True
    )
    return dataloader
