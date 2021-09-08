import sys
import torch
import numpy
import argparse
import torch.nn as nn

from attack import FWAdampAttackPlus
from attack import EnergyAttack

from model import get_model_for_attack
from eval_model import eval_model_with_attack
from utils import get_test_cifar, get_test_imagenet, get_test_mnist
from utils import print_attack_main_args

from torch.utils.data import DataLoader

torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Energy Attack benchmark.')
    parser.add_argument(
        '--batch_size', type=int, default=128, metavar='N',
        help='Batch size for training. Default: 128')
    parser.add_argument(
        '--subbatch_size', type=int, default=16, metavar='N',
        help='Sub-batch size for supported attacks (ta, energy). Default: 16')
    parser.add_argument(
        '--epsilon', type=float, default=0.05,
        help='max distance for attack. Default: 0.05')
    parser.add_argument(
        '--perturb_steps', type=int, default=10000,
        help='Maximum queries for attack. Default: 10000')
    parser.add_argument(
        '--model_name', type=str, default="model_vgg16bn",
        help='Model to attack. Default: model_vgg16bn')
    parser.add_argument(
        '--custom_flags', type=str, default=[], action='append',
        help="Customized flags for specific attacks"
    )
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument(
        '--step_size', type=float, default=0.003,
        help='Stepsize used in PGD-based attacks.'
        'This argument is not required if using FW or Energy.')
    parser.add_argument(
        '--attacker',
        choices=[
            'fw', 'energy',
        ],
        default='energy')
    parser.add_argument(
        '--dataset', choices=['cifar10', 'imagenet', 'mnist', 'rand_mnist'], default='imagenet',
        help='Dataset on which the attack is performed. Default: ImageNet')
    return parser.parse_args()


def merge_state_dicts(dicts):
    refd = dicts[0]
    result = dict()
    if isinstance(refd, dict):
        for k in refd.keys():
            result[k] = merge_state_dicts([d[k] for d in dicts])
        return result
    else:
        return sum(dicts)


def divide_state_dict(d, n):
    result = dict()
    if isinstance(d, dict):
        for k in d.keys():
            result[k] = divide_state_dict(d[k], n)
        return result
    else:
        return d / n


def get_attacker(attacker, step_size, epsilon, perturb_steps):
    if attacker == 'fw':
        print('Using FW-AdAmp', file=sys.stderr)
        return FWAdampAttackPlus(
            step_size, epsilon, perturb_steps)
    elif attacker == 'energy':
        print('Using Energy Attack', file=sys.stderr)
        return EnergyAttack(
            step_size, epsilon, perturb_steps)


class WrappedModel(nn.Module):
    def __init__(self, wrap, subbatch):
        super().__init__()
        self.wrap = wrap
        self.subbatch = subbatch

    def forward(self, x):
        subbatches = torch.split(x, self.subbatch)
        return torch.cat([self.wrap(sb) for sb in subbatches])


if __name__ == '__main__':
    args = parse_args()
    print_attack_main_args(args)
    device = torch.device(args.device)
    if (
        args.model_name.startswith('model')
    ):
        # load model to attack
        model = get_model_for_attack(args.model_name).to(device)
    else:
        raise ValueError(f'Model {args.model_name} does not exist.')
    model = WrappedModel(model, args.subbatch_size)
    # load attack method
    attack = get_attacker(
        args.attacker, args.step_size, args.epsilon,
        args.perturb_steps
    )
    model.eval()
    if args.dataset == 'cifar10':
        test_loader = get_test_cifar(args.batch_size)
    elif args.dataset == 'mnist':
        test_loader = get_test_mnist(args.batch_size)
    elif args.dataset == 'rand_mnist':
        test_loader = get_test_mnist(1)
        xs = [x for (x,), _ in test_loader]
        ys = numpy.load("rand_mnist.npy")
        ds = list(zip(xs, ys))
        test_loader = DataLoader(ds, args.batch_size)
    else:
        test_loader = get_test_imagenet(args.batch_size)
    # non-targeted attack
    natural_acc, robust_acc, distance = eval_model_with_attack(
        model, test_loader, attack, args.epsilon, device, args.dataset)
    print(
        "Natural Acc: %.5f, Robust acc: %.5f, distance: %.5f" %
        (natural_acc, robust_acc, distance)
    )
    if hasattr(attack, 'consumed_steps'):
        consumed_steps = numpy.array(attack.consumed_steps)
        consumed_steps = consumed_steps[consumed_steps <=
                                        attack.perturb_steps]
        print("Mean", numpy.mean(consumed_steps))
        print("Median", numpy.median(consumed_steps))
