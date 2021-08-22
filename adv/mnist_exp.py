from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
import tqdm
import torch.nn.functional as F


if __name__ == '__main__':
    dev = 'cuda:1'
    dnn = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28 * 3, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10),
    ).to(dev)
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
    ds = MNIST('./data', train=True, transform=trs, download=True)
    dl = DataLoader(ds, 100, True, num_workers=8)
    opt = torch.optim.Adam(list(dnn.parameters()) + list(cnn.parameters()))
    for epoch in range(50):
        prog = tqdm.tqdm(dl)
        for x, y in prog:
            y_cpu = y.detach().cpu().numpy()
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit_1 = dnn(x)
            logit_2 = cnn(x)
            loss = F.cross_entropy(logit_1, y) + F.cross_entropy(logit_2, y)
            loss.backward()
            prog.set_description(
                "Accs: %.4f, %.4f"
                % ((logit_1.detach().cpu().numpy().argmax(-1) == y_cpu).mean(),
                   (logit_2.detach().cpu().numpy().argmax(-1) == y_cpu).mean())
            )
            opt.step()
        torch.save({'dnn': dnn, 'cnn': cnn}, "mnist.pt")
