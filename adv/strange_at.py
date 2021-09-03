from torch import nn
from utils import get_test_cifar
from torchvision import transforms
import torch
import tqdm
import numpy
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from models import modified_wrn
from eval_model import eval_model_pgd


if __name__ == '__main__':
    dev = 'cuda:0'
    cnn = nn.Sequential(
        nn.Conv2d(3, 12, 5),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(12, 36, 5),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(5 * 5 * 36, 10)
    ).to(dev)
    cnn = torch.nn.DataParallel(cnn, [0])
    trs = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = list(CIFAR10('./data', train=True, transform=trs, download=True))
    x, y = [x for x, _ in ds], torch.LongTensor([y for _, y in ds])
    y = y[torch.randperm(len(y))]
    dl = DataLoader(list(zip(x, y)), 1000, True)
    opt = torch.optim.Adam(cnn.parameters())
    for epoch in range(500):
        prog = tqdm.tqdm(dl)
        accs = []
        losses = []
        for x, y in prog:
            y_cpu = y.detach().cpu().numpy()
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit_2 = cnn(x)
            loss = F.cross_entropy(logit_2, y)
            loss.backward()
            losses.append(loss.item())
            accs.append((logit_2.detach().cpu().numpy().argmax(-1) == y_cpu).mean())
            prog.set_description(
                "Epoch: %d, Loss: %.4f, Acc: %.4f" % (epoch, numpy.mean(losses), numpy.mean(accs))
            )
            opt.step()
        eval_model_pgd(cnn, DataLoader(list(dl)[:5], batch_size=None), dev, 4/255, 8/255, 10, 'cifar10')
        torch.save(cnn.module, "sat/exp_randperm_cifar_%d.pt" % epoch)
    raise Exception("Expected exception")
    cnn = torch.load('sat/exp_randperm_cifar_50.pt').to(dev)
    '''fc_param = list(cnn.fc.parameters())
    for p in cnn.parameters():
        if not any(p is p2 for p2 in fc_param):
            p.requires_grad = False'''
    cnn.fc = nn.Sequential(
        nn.Linear(640, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10)
    ).to(dev)
    opt = torch.optim.SGD([dict(
        params=cnn.fc.parameters(), lr=0.01, momentum=0.9
    ), dict(
        params=[p for p in cnn.parameters() if not any(p is q for q in cnn.fc.parameters())],
        lr=0.001, momentum=0.5
    )])
    cnn = torch.nn.DataParallel(cnn, [0, 1, 2])
    dl = DataLoader(ds, 300, True)
    test_dl = get_test_cifar(300)
    for epoch in range(100):
        prog = tqdm.tqdm(dl)
        accs = []
        losses = []
        for x, y in prog:
            y_cpu = y.detach().cpu().numpy()
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit_2 = cnn(x)
            loss = F.cross_entropy(logit_2, y)
            loss.backward()
            losses.append(loss.item())
            accs.append((logit_2.detach().cpu().numpy().argmax(-1) == y_cpu).mean())
            prog.set_description(
                "Epoch: %d, Loss: %.4f, Acc: %.4f" % (epoch, numpy.mean(losses), numpy.mean(accs))
            )
            opt.step()
        if epoch % 3 == 0:
            eval_model_pgd(cnn, list(test_dl)[:5], dev, 4/255, 8/255, 20, 'cifar10')
        torch.save(cnn.module, "exp_randperm_cifar_finetuned.pt")
