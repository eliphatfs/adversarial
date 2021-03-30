import argparse
import torch
import torch.nn as nn
from tqdm import trange


def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.007,
                        help='step size for pgd attack(default:0.03)')
    parser.add_argument('--perturb_steps', type=int, default=10,
                        help='iterations for pgd attack (default: pgd20)')
    parser.add_argument('--epsilon', type=float, default=8./255.,
                        help='max distance for pgd attack (default: 8/255)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='iterations for pgd attack (default: pgd20)')
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
    return parser.parse_args()


def train_epoch(model, args, train_loader, device, optimizer, epoch):
    model.train()
    corrects = 0
    data_num = 0
    loss_sum = 0
    with trange(len(train_loader.dataset)) as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            x, y = data.to(device), target.to(device)
            data_num += x.shape[0]
            optimizer.zero_grad()
            model.train()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            with torch.no_grad():
                model.eval()
                pred = torch.argmax(output, dim=1)
                corrects += (pred == y).float().sum()
            lossitem = loss.item()
            pbar.set_description(f"Train Epoch:{epoch}, Loss:{lossitem:.3f},"
                                 + f" acc:{corrects / float(data_num):.4f}, ")
            pbar.update(x.shape[0])
    acc = corrects / float(data_num)
    mean_loss = loss_sum / float(batch_idx+1)
    return acc,  mean_loss
