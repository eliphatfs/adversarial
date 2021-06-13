from eval_model import eval_model
from utils import get_test_cifar
from model import get_model_for_defense
import torch
import torch.nn as nn
import argparse


def parse_infer_args():
    parser = argparse.ArgumentParser(
        description='Basic Infer on CIFAR-10 without Attack')
    parser.add_argument(
        '--batch_size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--model_name', type=str, default="")
    parser.add_argument(
        '--model',
        choices=['ResNet18', 'PreActResNet18',
                 'ResNet34', 'PreActResNet34', 'WideResNet28'],
        default='')
    parser.add_argument(
        '--model_path', type=str,
        default=''
    )
    parser.add_argument('--device', type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_infer_args()
    device = torch.device(args.device)

    model = get_model_for_defense(args.model_name).to(device)
    model = nn.DataParallel(model, device_ids=[0])

    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    correct = []
    num = 0
    natural_acc, _ = eval_model(model, test_loader, args.device)
    print(
        f"Natual Acc: {natural_acc:.5f}"
    )
