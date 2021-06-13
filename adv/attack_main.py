import torch
import torch.nn as nn
import sys
import argparse

from utils import get_test_cifar
from attack import ArchTransferAttack
from attack import BarrierMethodAttack
from attack import BetterSecondOrderAttack
from attack import ChihaoHappyAttack
from attack import DeepFoolAttack
from attack import FWAdampAttackPlus
from attack import PGDAttack
from attack import SobolHappyAttack
from model import get_custom_model, get_model_for_attack, get_model_for_defense
from eval_model import eval_model_with_attack


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                        help='step size for pgd attack(default:0.003)')
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                        help='max distance for pgd attack (default: 8/255)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_name', type=str, default="")
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
    parser.add_argument(
        '--attacker',
        choices=[
            'pgd', 'fw', 'arch_transfer', 'barrier',
            'stochastic_sample', 'sobol_sample',
            'deepfool', 'second_order'
        ],
        default='fw')
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
    elif attacker == 'pgd':
        print('Using PGD', file=sys.stderr)
        return PGDAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'arch_transfer':
        print('Using Arch Transfer', file=sys.stderr)
        return ArchTransferAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'barrier':
        print('Using Barrier Method Attack', file=sys.stderr)
        return BarrierMethodAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'stochastic_sample':
        print('Using Random Sampling', file=sys.stderr)
        return ChihaoHappyAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'sobol_sample':
        print('Using Sobol Sampling', file=sys.stderr)
        return SobolHappyAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'deepfool':
        print('Using DeepFool', file=sys.stderr)
        return DeepFoolAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'second_order':
        print('Using Second Order Attack', file=sys.stderr)
        return BetterSecondOrderAttack(
            step_size, epsilon, perturb_steps)


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    if 'model' in args.model_name:
        model = get_model_for_attack(args.model_name).to(device)
        # 根据model_name, 切换要攻击的model
    elif args.model_name != '':
        model = get_model_for_defense(args.model_name).to(device)
    else:
        model = get_custom_model(args.model, args.model_path).to(device)
    # 攻击任务：Change to your attack function here
    # Here is a attack baseline: PGD attack
    model = nn.DataParallel(model, device_ids=[0])
    attack = get_attacker(
        args.attacker, args.step_size, args.epsilon, args.perturb_steps)
    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc, robust_acc, distance = eval_model_with_attack(
        model, test_loader, attack, args.epsilon, device)
    print(
        "Natural Acc: %.5f, Robust acc: %.5f, distance: %.5f" %
        (natural_acc, robust_acc, distance)
    )
