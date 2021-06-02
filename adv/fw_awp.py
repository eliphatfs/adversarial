import os
import sys
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from tqdm import trange

from eval_model import eval_model_pgd
from utils import prepare_cifar, Logger, check_mkdir
from eval_model import eval_model
from models.resnet import ResNet34, ResNet18
from models.preactresnet import PreActResNet18, PreActResNet34
from models.trades_wide_resnet import WideResNet28, WideResNet34
from experimental_attack import ExpAttack
from pgd_attack import PGDAttack
from awp import AdvWeightPerturb


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=float, default=0.007,
                        help='step size for pgd attack(default:0.007)')
    parser.add_argument('--perturb_steps', type=int, default=10,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--epsilon', type=float, default=8./255.,
                        help='max distance for pgd attack (default: 8/255)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='iterations for pgd attack (default pgd20)')
    # parser.add_argument('--lr_steps', type=str, default=,
    #                help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--epoch', type=int, default=100,
                        help='epochs for pgd training ')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay ratio')
    parser.add_argument('--adv_train', type=int, default=1,
                        help='If use adversarial training')
    parser.add_argument('--gpu_id', type=str, default="0,1")
    parser.add_argument('--awp_gamma', type=float, default=0.01)
    parser.add_argument('--awp_warmup', type=int, default=0)
    parser.add_argument('--attacker', choices=['pgd', 'fw'], default='pgd')
    parser.add_argument(
        '--model',
        choices=['ResNet18', 'PreActResNet18',
                 'ResNet34', 'PreActResNet34', 'WideResNet28'],
        default='ResNet18')
    parser.add_argument('--model_name', default='fwawp-resnet')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--checkpoint_path', default='./pretrained/ckpt.pt')
    parser.add_argument('--checkpoint_start', type=int, default=50)
    parser.add_argument('--trades', type=bool, default=False)
    parser.add_argument('--trades_param', type=float,
                        default=6.0)  # 1/lambda in TRADES
    return parser.parse_args()


def train_fwawp_epoch(
        model: nn.Module,
        perturbator: AdvWeightPerturb,
        attacker,
        train_loader,
        device,
        optimizer: optim.Optimizer,
        epoch,
        args):
    """Frank-Wolfe w/ Adversarial Weight Perturbation.
    T.H.E. Ultimate 缝合怪
    """
    model.train()
    corrects_adv, corrects = 0, 0
    data_num = 0
    loss_sum = 0
    if epoch == args.awp_warmup:
        print('Activating AWP.', file=sys.stderr)

    with trange(len(train_loader.dataset)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            x, y = data.to(device), target.to(device)
            data_num += x.shape[0]

            # get adversarial examples
            x_adv = attacker(model, x, y)

            model.train()
            # add pertubation
            if epoch >= args.awp_warmup:
                if not args.trades:
                    perturbator.calc_awp(x_adv, y)
                else:
                    perturbator.calc_trades_awp(x_adv, x, y, args)
                perturbator.perturb_model()

            # gradient descent
            if not args.trades:
                # use standard cross entropy loss
                output_adv = model(x_adv)
                loss = nn.CrossEntropyLoss()(output_adv, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # use TRADE loss
                if epoch == 0:
                    print("Using TRADES loss.", file=sys.stderr)
                output_orig = model(x)
                output_adv = model(x_adv)
                loss_reg = F.kl_div(
                    F.log_softmax(output_orig, dim=1),
                    F.softmax(output_adv, dim=1),
                    reduction='batchmean')
                loss_orig = F.cross_entropy(output_orig, y)
                loss = loss_orig + args.trades_param * loss_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # restore awp
            if epoch >= args.awp_warmup:
                perturbator.restore_perturb()

            loss_sum += loss.item()
            with torch.no_grad():
                model.eval()
                pred_adv = output_adv.argmax(dim=1)
                pred = torch.argmax(model(x), dim=1)
                corrects_adv += (pred_adv == y).float().sum()
                corrects += (pred == y).float().sum()
            pbar.set_description(
                f"Train Epoch:{epoch}, Loss:{loss.item():.3f}, " +
                f"acc:{corrects / float(data_num):.4f}, " +
                f"r_acc:{corrects_adv / float(data_num):.4f}"
            )
            pbar.update(x.shape[0])
    acc, adv_acc = corrects / float(data_num), corrects_adv / float(data_num)
    mean_loss = loss_sum / float(batch_idx+1)
    return acc, adv_acc, mean_loss


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_model(model_name, device):
    if model_name == 'ResNet18':
        print("Model: ResNet18", file=sys.stderr)
        return ResNet18().to(device), ResNet18().to(device)
    elif model_name == 'ResNet34':
        print("Model: ResNet34", file=sys.stderr)
        return ResNet34().to(device), ResNet34().to(device)
    elif model_name == 'PreActResNet18':
        print("Model: PreActivate ResNet18", file=sys.stderr)
        return PreActResNet18().to(device), PreActResNet18().to(device)
    elif model_name == 'PreActResNet34':
        print("Model: PreActivate ResNet 34")
        return PreActResNet34().to(device), PreActResNet34().to(device)
    elif model_name == 'WideResNet28':
        print("Model: Wide ResNet28", file=sys.stderr)
        return WideResNet28().to(device), WideResNet28().to(device)
    elif model_name == 'WideResNet34':
        print("Model: Wide ResNet34", file=sys.stderr)
        return WideResNet34().to(device), WideResNet34().to(device)


def resume_training(
        checkpoint_path,
        model,
        model_optim: optim.Optimizer,
        proxy_optim: optim.Optimizer):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.module.load_state_dict(checkpoint['model'])
        model_optim.load_state_dict(checkpoint['model_optim'])
        proxy_optim.load_state_dict(checkpoint['proxy_optim'])
    except BaseException as err:
        print("? Failed to load models. Quitting...")
        print(err)
        quit()

    return model, model_optim, proxy_optim, checkpoint['epoch']


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_num = max(len(args.gpu_id.split(',')), 1)

    model_name = args.model_name
    log_dir = "logs/%s_%s" % (time.strftime("%b%d-%H%M",
                                            time.localtime()), model_name)
    check_mkdir(log_dir)
    log = Logger(log_dir+'/train.log')
    log.print(args)

    device = torch.device('cuda')
    model, proxy = get_model(args.model, device)
    model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    proxy = nn.DataParallel(proxy, device_ids=[i for i in range(gpu_num)])

    train_loader, test_loader = prepare_cifar(
        args.batch_size, args.test_batch_size)
    model_optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    proxy_optimizer = optim.SGD(
        proxy.parameters(),
        lr=0.01,)
    resume_epoch = 0

    if args.resume:
        model, model_optimizer, proxy_optimizer, resume_epoch =\
            resume_training(
                args.checkpoint_path,
                model,
                model_optimizer,
                proxy_optimizer)

    if args.attacker == 'pgd':
        attacker = PGDAttack(args.step_size, args.epsilon, args.perturb_steps)
    else:
        attacker = ExpAttack(args.step_size, args.epsilon, args.perturb_steps)
    perturbator = AdvWeightPerturb(
        model=model,
        proxy=proxy,
        proxy_optim=proxy_optimizer,
        gamma=args.awp_gamma,)

    best_epoch, best_robust_acc = 0, 0.
    losses = []
    test_accs = []
    test_robust_accs = []
    total_time = 0
    for e in range(resume_epoch, args.epoch):
        start_time = time.time()
        adjust_learning_rate(model_optimizer, e)

        # adv training
        train_acc, train_robust_acc, loss = train_fwawp_epoch(
            model, perturbator, attacker, train_loader,
            device, model_optimizer, e, args)

        losses.append(loss)

        # eval
        if e % 3 == 0 or (e >= 74 and e <= 80):
            test_acc, test_robust_acc, _ = eval_model_pgd(
                model, test_loader, device,
                args.step_size, args.epsilon, args.perturb_steps
            )
            test_accs.append(test_acc.item())
            test_robust_accs.append(test_robust_acc.item())
        else:
            test_acc, _ = eval_model(model,  test_loader, device)
            test_accs.append(test_acc.item())
        if test_robust_acc > best_robust_acc:
            best_robust_acc, best_epoch = test_robust_acc, e
            best_checkpoint = {
                'epoch': e,
                'model': model.module.state_dict(),
                'model_optim': model_optimizer.state_dict(),
                'proxy_optim': proxy_optimizer.state_dict(), }

        # save model & logging
        if e > args.checkpoint_start:
            torch.save({
                'epoch': e,
                'model': model.module.state_dict(),
                'model_optim': model_optimizer.state_dict(),
                'proxy_optim': proxy_optimizer.state_dict(), },
                os.path.join(
                    log_dir,
                    f"{model_name}-checkpoint.pt")
            )
            torch.save(best_checkpoint, os.path.join(
                log_dir, f"{model_name}-best.pt"))

        end_time = time.time()
        time_elapsed = end_time - start_time
        total_time += time_elapsed

        log.print(f"Epoch:{e}, loss:{loss:.5f}, train_acc:{train_acc:.4f}, train_robust_acc:{train_robust_acc:.4f}, " +
                  f"test_acc:{test_acc:.4f}, test_robust_acc:{test_robust_acc:.4f}, " +
                  f"best_robust_acc:{best_robust_acc:.4f} in epoch {best_epoch}, " +
                  f"epoch_time: {time_elapsed}, total_time_elapsed: {total_time}")
    torch.save(
        model.module.state_dict(),
        f"{log_dir}/{model_name}_e{args.epoch - 1}_{test_acc:.4f}_{test_robust_acc:.4f}-final.pt")
    pickle.dump(
        [test_accs, test_robust_accs, losses],
        open(os.path.join(log_dir, f'{model_name}_footprints.pkl'), 'wb'))
