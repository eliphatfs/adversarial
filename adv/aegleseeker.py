import torch
import torch.nn as nn
import torch.nn.functional as F


torch.square = lambda x: x ** 2


class AegleSeeker(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def reg(self):
        v = 0.0
        for module in self.wrapped.modules():
            if isinstance(module, nn.Linear):
                v = v + 0.0001 * torch.square(module.weight.reshape(-1)).sum()
            if isinstance(module, nn.BatchNorm2d):
                v = v + 0.0001 * torch.square(module.weight.reshape(-1)).sum()
            if isinstance(module, nn.Conv2d):
                v = v + 0.0001 * torch.square(module.weight.reshape(-1)).sum()
                s = list(module.weight.shape)
                s[0] = s[0] * s[1]
                s[1] = 1
                lagrange = F.conv2d(module.weight.reshape(*s), module.weight.new_tensor([
                    [+0, +1, +0],
                    [+1, -4, +1],
                    [+0, +1, +0]
                ]).reshape(1, 1, 3, 3), padding=1)
                v = v - 0.5 * (lagrange * module.weight.reshape(*s)).reshape(-1).sum()
        return v

    def forward(self, x):
        return self.wrapped(x)
