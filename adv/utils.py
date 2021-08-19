import torch
import torchvision
from torchvision import transforms

import os
import sys


def print_attack_main_args(args):
    print("===== Arguments =====")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Step size: {args.step_size}")
    print(f"  - Epsilon: {args.epsilon:.5f}")
    print(f"  - Perturb steps: {args.perturb_steps}")
    print(f"  - Model name: {args.model_name}")
    print(f"  - Model architecture: {args.model if args.model != '' else args.model_name}")
    print(f"  - Attacker: {args.attacker}")
    print(f"  - Targeted: {args.targeted == 'targeted'}")


def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Logger:
    def __init__(self, file, print=True):
        self.file = file
        # local_time = time.strftime("%b%d_%H%M%S", time.localtime())
        # self.file += local_time
        self.All_file = 'logs/All.log'

    def print(self, content='', end='\n', file=None):
        if file is None:
            file = self.file
        with open(file, 'a') as f:
            if isinstance(content, str):
                f.write(content+end)
            else:
                old = sys.stdout
                sys.stdout = f
                print(content)
                sys.stdout = old
        if file is None:
            self.print(content, file=self.All_file)
        print(content, end=end)


def get_test_cifar(batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    return test_loader


def prepare_cifar(batch_size, test_batch_size):
    kwargs = {'num_workers': 8, 'pin_memory': True}
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
