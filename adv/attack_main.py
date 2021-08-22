import torch
import torch.nn as nn
import sys
import argparse
import numpy

from utils import get_test_cifar, get_test_imagenet, get_test_mnist, print_attack_main_args
from attack import ArchTransferAttack
from attack import BarrierMethodAttack
from attack import BetterSecondOrderAttack
from attack import ChihaoHappyAttack
from attack import DeepFoolAttack
from attack import FWAdampAttackPlus
from attack import PGDAttack
from attack import SobolHappyAttack
from attack import EnergyAttack
try:
    from attack.torchattackext import TAEXT
except ImportError:
    pass
try:
    from mnist_vit import MegaSizer
except ImportError:
    pass
from models import WideResNet
from model import get_model_for_attack
from models import WideResNet28
from eval_model import eval_model_with_attack
from attack import StochasticFWAdampAttack
from attack import SPSA
from model import get_custom_model, get_model_for_defense
from eval_model import eval_model_with_targeted_attack


torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--subbatch_size', type=int, default=16, metavar='N',
                        help='sub batch size for supported attacks (ta, energy)')
    parser.add_argument('--step_size', type=int, default=0.003,
                        help='step size for pgd attack(default:0.003)')
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                        help='max distance for pgd attack (default: 8/255)')
    parser.add_argument('--perturb_steps', type=int, default=10000,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_name', type=str, default="model1")
    parser.add_argument(
        '--custom_flags', type=str, default=[], action='append',
        help="Customized flags for specific attacks"
    )
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
            'deepfool', 'second_order', 'energy', 'ta', 'stoch_fw', 'spsa'
        ],
        default='energy')
    parser.add_argument(
        '--targeted', choices=['targeted', 'untargeted'], default='untargeted')
    parser.add_argument(
        '--dataset', choices=['cifar10', 'imagenet', 'mnist'], default='cifar10')
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
    elif attacker == 'energy':
        print('Using Energy Attack', file=sys.stderr)
        return EnergyAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'ta':
        print('Using External Attack', file=sys.stderr)
        return TAEXT(step_size, epsilon, perturb_steps)
    elif attacker == 'stoch_fw':
        print('Using SPSA FW-AdAmp')
        return StochasticFWAdampAttack(
            step_size, epsilon, perturb_steps)
    elif attacker == 'spsa':
        print('Using SPSA')
        return SPSA(
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
        model = get_model_for_attack(args.model_name).to(device)
        # 根据model_name, 切换要攻击的model
    elif args.model_name != '':
        model = get_model_for_defense(args.model_name).to(device)
    else:
        model = get_custom_model(args.model, args.model_path).to(device)
    # 攻击任务：Change to your attack function here
    # Here is a attack baseline: PGD attack
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model = WrappedModel(model, args.subbatch_size)
    attack = get_attacker(
        args.attacker, args.step_size, args.epsilon,
        args.perturb_steps
    )
    model.eval()
    if args.dataset == 'cifar10':
        test_loader = get_test_cifar(args.batch_size)
    elif args.dataset == 'mnist':
        test_loader = get_test_mnist(args.batch_size)
    else:
        test_loader = get_test_imagenet(args.batch_size)
    if args.targeted == 'untargeted':
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
    else:
        if args.attacker != 'pgd' and args.attacker != 'fw':
            raise NotImplementedError(
                f"Targeted attack of {args.attacker} is currently pigeoned.")
        # targeted attack, default target is 0
        natural_acc, robust_acc, success_rate, distance =\
            eval_model_with_targeted_attack(
                model, test_loader, attack, args.epsilon, device, args.dataset)
        print(
            "Natural Acc: %.5f, Robust acc: %.5f, Success Rate: %.5f, distance: %.5f" %
            (natural_acc, robust_acc, success_rate, distance)
        )
