import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import sys
import pandas as pd
from torchvision.datasets.mnist import MNIST


def print_attack_main_args(args):
    print("===== Arguments =====")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Step size: {args.step_size}")
    print(f"  - Epsilon: {args.epsilon:.5f}")
    print(f"  - Perturb steps: {args.perturb_steps}")
    print(
        "  - Model name:"
        f"{args.model_path if args.model_path != '' else args.model_name}")
    print(
        "  - Model architecture: "
        f"{args.model if args.model != '' else args.model_name}")
    print(f"  - Dataset: {args.dataset}")
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
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_test_mnist(batch_size):
    trs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x] * 3))
    ])
    ds = MNIST('./data', train=False, transform=trs, download=True)
    dl = DataLoader(ds, batch_size, False)
    return dl


class ImageSet(Dataset):
    def __init__(self, df, input_dir, transformer):
        self.df = df
        self.transformer = transforms.Compose([
            transforms.Scale(299),
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ])
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_name = self.df.iloc[item]['image_path']
        image_path = os.path.join(self.input_dir, image_name)
        image = self.transformer(Image.open(image_path).convert("RGB"))
        label_idx = self.df.iloc[item]['label_idx']
        # target_idx = self.df.iloc[item]['target_idx']
        sample = [
            # 'dataset_idx': item,
            image,  # image
            label_idx,  # label
            # 'target': target_idx+1,
            # 'filename': image_name
        ]
        return sample


def get_test_imagenet(batch_size):
    input_dir = './data/images2/'
    dev_data = pd.read_csv(
        input_dir + 'new_val_label.txt',
        header=None, sep=' ',
        names=['image_path', 'label_idx', 'target_idx'])
    transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                    std=[0.5, 0.5, 0.5]),
    ])
    datasets = ImageSet(dev_data, input_dir, transformer)
    dataloader = DataLoader(datasets,
                            batch_size=batch_size,
                            shuffle=False)
    return dataloader


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
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


def get_dataset_size(dataset):
    if dataset == 'imagenet':
        return 1000
    else:
        return 10000
