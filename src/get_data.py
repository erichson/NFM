import os

import numpy as np
import torch
from PIL import Image
from robustness.tools.folder import ImageFolder
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset

from torchvision.transforms import ToPILImage  # built-in function

IMAGENET_TRAIN_FOLDER = '/scratch/data/imagenet12/train'
IMAGENET_TEST_FOLDER = '/scratch/data/imagenet12/val'

# download from https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
CIFAR10C_FOLDER = 'data/cifar/CIFAR-10-C/'
NOISE_TYPES = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]
SEVERITIES = [1, 2, 3, 4, 5]


class CIFAR10Corrupt(VisionDataset):

    def __init__(self,
                 root="data/CIFAR-10-C",
                 severity=[1, 2, 3, 4, 5],
                 noise=None,
                 transform=None,
                 target_transform=None):
        super(CIFAR10Corrupt, self).__init__(root, transform=transform, target_transform=target_transform)

        noise = NOISE_TYPES if noise is None else noise

        X = []
        for n in noise:
            D = np.load(os.path.join(root, f"{n}.npy"))
            D_s = np.split(D, 5, axis=0)
            for s in severity:
                X.append(D_s[s - 1])
        X = np.concatenate(X, axis=0)
        Y = np.load(os.path.join(root, "labels.npy"))
        Y_s = np.split(Y, 5, axis=0)
        Y = np.concatenate([Y_s[s - 1] for s in severity])
        Y = np.repeat(Y, len(noise))

        self.data = X
        self.targets = Y
        self.noise_to_nsamples = (noise, X.shape, Y.shape)
        print(f"Loaded {severity}-{noise}: X {X.shape} Y: {Y.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        #img = ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target





def getData(name='cifar10', train_bs=128, test_bs=512, train_path=None, test_path=None, severity=0, noise='fog'):

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)

        testset = datasets.CIFAR10(root='../cifar10', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)

    
    
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = datasets.CIFAR100(root='../cifar100', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)

        testset = datasets.CIFAR100(root='../cifar100', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)

    elif name == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = ImageFolder(root=IMAGENET_TRAIN_FOLDER, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=train_bs,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=32,
                                                   prefetch_factor=2)

        test_set = ImageFolder(root=IMAGENET_TEST_FOLDER, transform=transform_train)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=test_bs,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=32)

    elif name == 'cifar10c':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True,
                                                   num_workers=4,
                                                   pin_memory=True)

        testset = CIFAR10Corrupt(root=CIFAR10C_FOLDER,
                                          severity=[severity],
                                          noise=[noise],
                                          transform=transform_test)
        
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  pin_memory=True)

    return train_loader, test_loader

