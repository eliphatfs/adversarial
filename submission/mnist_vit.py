import torch
import tqdm
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import InterpolationMode

from pytorch_pretrained_vit import ViT


class MegaSizer(torch.nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap
        self.sizer = transforms.Resize(
            224, interpolation=InterpolationMode.BICUBIC)

    def forward(self, x):
        x = self.sizer(x)
        return self.wrap(x)


if __name__ == '__main__':
    dev = 'cuda:1'
    mnist_models = torch.load("mnist.pt")
    model = MegaSizer(ViT('B_16', pretrained=True)).to(dev)

    trs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x] * 3))
    ])
    ds = MNIST('./data', train=True, transform=trs, download=True)
    dl = DataLoader(ds, 32, True, num_workers=8)
    opt = torch.optim.Adam(model.parameters(), eps=1e-3)
    for epoch in range(20):
        prog = tqdm.tqdm(dl)
        for x, y in prog:
            y_cpu = y.detach().cpu().numpy()
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logit = model(x)
            loss = F.cross_entropy(logit, y)
            loss.backward()
            prog.set_description(
                "Accs: %.4f"
                % ((logit.detach().cpu().numpy().argmax(-1) == y_cpu).mean())
            )
            opt.step()
        mnist_models['vit'] = model
        torch.save(mnist_models, "mnist.pt")
