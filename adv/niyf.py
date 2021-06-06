import torch


class NIYFModel(torch.nn.Module):

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, x):
        mer = 0
        for _ in range(5):
            inner = self.wrapped(x + torch.randn_like(x) * (8 / 255))
            mer = mer + torch.softmax(inner, -1)
        return mer
