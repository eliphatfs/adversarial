import torch

from utils import get_test_cifar
from pgd_attack import PGDAttack
from models import WideResNet
from model import get_model_for_attack
from eval_model import eval_model_with_attack
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
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
    parser.add_argument('--gpu_id', type=str, default="0,1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 多卡机设置使用的gpu卡号
    gpu_num = max(len(args.gpu_id.split(',')), 1)
    device = torch.device('cpu')
    # model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    if args.model_name != "":
        model = get_model_for_attack(args.model_name).to(device)
        # 根据model_name, 切换要攻击的model
    else:
        # 防御任务, Change to your model here
        model = WideResNet()
        model.load_state_dict(torch.load(
            'models/weights/wideres34-10-pgdHE.pt'))
    # 攻击任务：Change to your attack function here
    # Here is a attack baseline: PGD attack
    attack = PGDAttack(args.step_size, args.epsilon, args.perturb_steps)
    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc, robust_acc, distance = eval_model_with_attack(
        model, test_loader, attack, device)
    print(
        "Natural Acc: %.5f, Robust acc: %.5f, distance: %.5f" %
        (natural_acc, robust_acc, distance)
    )
