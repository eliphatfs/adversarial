from torch import nn
from torchvision import transforms
import torch
import tqdm
import numpy
from utils import get_test_mnist
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F


if __name__ == '__main__':
    dev = 'cuda:0'
    cnn = nn.Sequential(
        nn.Conv2d(3, 12, 5),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(12, 36, 5),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 36, 10)
    ).to(dev)
    trs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x] * 3))
    ])
    ds = list(MNIST('./data', train=True, transform=trs, download=True))
    x, y = [x for x, _ in ds], torch.LongTensor([y for _, y in ds])
    y = y[torch.randperm(len(y))]
    dl = DataLoader(list(zip(x, y)), 500, True)
    opt = torch.optim.Adam(cnn.parameters())
    for epoch in range(30):
        prog = tqdm.tqdm(dl)
        for x, y in prog:
            y_cpu = y.detach().cpu().numpy()
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit_2 = cnn(x)
            loss = F.cross_entropy(logit_2, y)
            loss.backward()
            prog.set_description(
                "Accs: %.4f"
                % ((logit_2.detach().cpu().numpy().argmax(-1) == y_cpu).mean())
            )
            opt.step()
        torch.save(cnn, "exp_randperm_mnist_cnn.pt")
    ys = []
    with torch.no_grad():
        for x, y in tqdm.tqdm(get_test_mnist(100)):
            x, y = x.to(dev), y.to(dev)
            logit = cnn(x).cpu().numpy()
            ys.extend(logit.argmax(-1))
    numpy.save("rand_mnist.npy", ys)
