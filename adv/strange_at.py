from utils import get_test_cifar
from torchvision import transforms
import torch
import tqdm
import numpy
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from models import ModifiedWRN28
from eval_model import eval_model_pgd


if __name__ == '__main__':
    dev = 'cuda:0'
    cnn = ModifiedWRN28().to(dev)
    trs = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = list(CIFAR10('./data', train=True, transform=trs, download=True))
    x, y = [x for x, _ in ds], torch.LongTensor([y for _, y in ds])
    y = y[torch.randperm(len(y))]
    dl = DataLoader(list(zip(x, y)), 100, True)
    opt = torch.optim.Adam(cnn.parameters())
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
        torch.save(cnn, "exp_randperm_cifar_wrn28.pt")
    opt = torch.optim.Adam(cnn.fc.parameters(), eps=1e-2)
    dl = DataLoader(ds, 100, True)
    test_dl = get_test_cifar(100)
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
        eval_model_pgd(cnn, test_dl, dev, 4/255, 8/255, 20, 'cifar10')
        torch.save(cnn, "exp_randperm_cifar_wrn28.pt")
