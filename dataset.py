import os
from copy import deepcopy

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets


def load_mnist(is_train_set=True, with_jitter=False):
    transform_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
    if with_jitter:
        transform_list.append(torchvision.transforms.GaussianBlur(3, 0.5))
    dset = datasets.MNIST(
        os.path.dirname(os.path.realpath(__file__)) + "/data",
        train=is_train_set,
        download=True,
        transform=torchvision.transforms.Compose(transform_list),
    )
    return dset


def get_dataloader(dataset, batch_size=100):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return dataloader


def filter_by_class(dset, classes):
    indices = dset.targets == classes[0]
    for c in classes[1:]:
        indices |= dset.targets == c
    filtered_dset = deepcopy(dset)
    filtered_dset.data, filtered_dset.targets = filtered_dset.data[indices], filtered_dset.targets[indices]
    return filtered_dset
