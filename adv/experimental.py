import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import itertools


def get_train_data():
    trainset_org = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )
    loader_org = torch.utils.data.DataLoader(trainset_org, 100)
    return itertools.chain(loader_org)


class NNAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        with torch.no_grad():
            model(x)
            x_bu = x
            x = x.flatten(1)
            x = x - x.mean(1, keepdim=True)
            x = x / torch.norm(x, dim=-1, keepdim=True)
            proj_x = torch.matmul(x, model.projector[:, :30])
            x = x.reshape(*x_bu.shape)
            cdist = torch.cdist(
                model.data, proj_x.unsqueeze(0)).squeeze(0)  # p * n
            indices = torch.argsort(cdist, dim=0)[:100]
            raw_labels = model.raw_labels[indices.reshape(
                -1)].reshape(*indices.shape)
            indices = indices[raw_labels != y[0]]
            return x + self.epsilon * torch.sign(model.raw[indices[0].item()].reshape(*x.shape) - x)


class KNN(nn.Module):
    def __init__(self):
        super().__init__()
        train_loader = get_train_data()
        data = [(x, y) for x, y in train_loader]
        xs = torch.cat([x for x, y in data]).flatten(1)
        xs = xs - torch.mean(xs)
        xs = xs / torch.norm(xs, dim=-1, keepdim=True)
        self.inited = False
        self.raw = nn.Parameter(xs)
        self.raw_labels = torch.cat([y for x, y in data])
        self.labels = nn.Parameter(nn.functional.one_hot(
            self.raw_labels, 10).to(self.raw.dtype))

    def init(self):
        self.inited = True
        with torch.no_grad():
            _, _, v = torch.svd(self.raw)
            self.projector = v
            self.data = torch.matmul(self.raw, v[:, :30]).unsqueeze(0)

    def pairwise_dist(self, x, y):
        # x (*, P, D)
        # y (*, R, D)
        dotted = torch.bmm(x, y.transpose(-1, -2))  # BPR
        return dotted / (
            torch.norm(x, dim=-1, keepdim=True)
            * torch.norm(y, dim=-1, keepdim=True).transpose(-1, -2)
            + 1e-8
        )

    def forward(self, x):
        if not self.inited:
            self.init()
        x = x.flatten(1)
        x = x - x.mean(1, keepdim=True)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        proj_x = torch.matmul(x, self.projector[:, :30])
        cdist = torch.cdist(self.data, proj_x.unsqueeze(0)).squeeze(0)  # p * n
        _, indices = torch.topk(cdist, 15, dim=0, largest=False)  # k * n
        gdist = torch.gather(cdist, 0, indices)  # k * n
        # k * n * 10
        labels = self.labels[indices.reshape(-1)].reshape(*indices.shape, 10)
        return torch.log((labels * torch.exp(-gdist.unsqueeze(-1))).sum(0))
