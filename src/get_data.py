import torch
from torchvision import datasets, transforms
from robustness.tools.folder import ImageFolder

IMAGENET_TRAIN_FOLDER = '/scratch/data/imagenet12/train'
IMAGENET_TEST_FOLDER = '/scratch/data/imagenet12/val'

def getData(name='cifar10', train_bs=128, test_bs=512, train_path=None, test_path=None):    
    
    if name == 'cifar10':
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
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=False)

        testset = datasets.CIFAR10(root='../cifar10', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=False)
    
    
    
    elif name == 'cifar100':
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

        trainset = datasets.CIFAR100(root='../cifar100', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True, num_workers=4, pin_memory=False)

        testset = datasets.CIFAR100(root='../cifar100', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=4, pin_memory=False)

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
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs, 
                                    shuffle=True, pin_memory=True, num_workers=32,
                                    prefetch_factor=2)

        test_set = ImageFolder(root=IMAGENET_TEST_FOLDER, transform=transform_train)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bs, 
                                    shuffle=False, pin_memory=True, num_workers=32)

    else:
        raise NameError('dataset is not supported')
        
    return train_loader, test_loader







