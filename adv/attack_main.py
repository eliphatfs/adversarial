# %%

import torch
import torch.nn as nn

from utils import get_test_cifar
from pgd_attack import PGDAttack
from pnewton_test import PNewtonAttack
from mod_pgd_attack import PGDAttackMod2
from chihao_happy_attack import ChihaoHappyAttack
from sobol_happy_attack import SobolHappyAttack
from hybrid_attack import HybridAttack
from second_order_attack import SecondOrderAttack
from better_second_order_attack import BetterSecondOrderAttack
from deep_fool import DeepFoolAttack
from experimental_attack import ExpAttack
from spsa import SPSA
from models import WideResNet
from model import get_model_for_attack
from eval_model import eval_model_with_attack
from arch_transfer_attack import ArchTransferAttack
import argparse
from barrier_attack import BarrierMethodAttack
import vgg


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                        help='step size for pgd attack(default:0.003)')
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                        help='max distance for pgd attack (default: 8/255)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument(
        '--model_path', type=str,
        default="./models/weights/model-wideres-pgdHE-wide10.pt"
    )
    parser.add_argument('--device', type=str, default="cpu")
    return parser.parse_args()


'''for i in range(0, 7):
    model = get_model_for_attack('model' + str(i)).cpu()
    import torch.jit
    trace_module = torch.jit.trace(model, torch.randn([1, 3, 32, 32]))
    del trace_module.training
    trace_module.eval()
    frozen_module = torch.jit.freeze(trace_module)
    code, consts = frozen_module.code_with_constants
    with open("%s.artifact.py" % ('model' + str(i)), 'w') as fo:
        fo.write(code)'''


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


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)
    if args.model_name != "":
        model = get_model_for_attack(args.model_name).to(device)
        # 根据model_name, 切换要攻击的model
    else:
        # 防御任务, Change to your model here
        model = vgg.vgg13_bn().to(device)
        ensemble = torch.load('vgg13bn_regm1.dat', map_location=torch.device(device))
        '''for i in range(1, 7):
            ensemble = merge_state_dicts([
                ensemble,
                torch.load('vgg13bn_regm%d.dat' % i, map_location=torch.device(device))
            ])
        ensemble = divide_state_dict(ensemble, 7)'''
        model.load_state_dict(ensemble)
    # 攻击任务：Change to your attack function here
    # Here is a attack baseline: PGD attack
    # model = nn.DataParallel(model, device_ids=[0, 4, 6, 7])
    attack = PGDAttack(args.step_size, args.epsilon, args.perturb_steps)
    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc, robust_acc, distance = eval_model_with_attack(
        model, test_loader, attack, args.epsilon, device)
    print(
        "Natural Acc: %.5f, Robust acc: %.5f, distance: %.5f" %
        (natural_acc, robust_acc, distance)
    )
