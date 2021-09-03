from utils import prepare_cifar, get_test_cifar
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import eval_model
import math
from attack.energy_attack import EnergyAttack


def patches_zero_mean(bchw, kernel=5, stride=1, pad_same=True):
    b, c, h, w = bchw.shape
    if pad_same:
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_h = (h2 - 1) * stride + (kernel - 1) + 1 - h
        pad_w = (w2 - 1) * stride + (kernel - 1) + 1 - w
        x = F.pad(bchw, (pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2))
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    
    patches = patches.view(b, -1, patches.shape[-2], patches.shape[-1])
    return patches - patches.mean([-1, -2, -3], keepdim=True)


class MinMaxPool(nn.Module):
    def __init__(self, maxpool_gen):
        super().__init__()
        self.mp1 = maxpool_gen()
        self.mp2 = maxpool_gen()

    def forward(self, x):
        return torch.cat([self.mp1(x), -self.mp2(-x)], 1)


class FixedPCA(nn.Module):
    def __init__(self, coef, kernel):
        super().__init__()
        self.coef = nn.Parameter(coef, False)
        self.kernel = kernel

    def forward(self, x):
        patch = patches_zero_mean(x, self.kernel)
        return torch.einsum("bihw,ij->bjhw", patch, self.coef)


def next_pca_weights(image_loader, tn_repr, kernel, in_feat, out_feat):
    cov = torch.zeros([kernel ** 2 * in_feat] * 2).to(tn_repr)
    for x in image_loader:
        patch = patches_zero_mean(x.to(tn_repr), kernel)
        cov = cov + torch.einsum('bphw,bqhw->pq', patch, patch)
    sigma, eigv = torch.linalg.eigh(cov)
    return eigv[:, -out_feat:]


class WrappedModel(nn.Module):
    def __init__(self, wrap, subbatch):
        super().__init__()
        self.wrap = wrap
        self.subbatch = subbatch

    def forward(self, x):
        subbatches = torch.split(x, self.subbatch)
        return torch.cat([self.wrap(sb) for sb in subbatches])


if __name__ == '__main__':
    train, test = prepare_cifar(400, 400)
    tn_repr = torch.zeros([3]).to('cuda:1')
    train_x = [x for x, _ in train]
    encoder_config = [
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 5, 3, 64), 5),
        lambda _: nn.Tanh(),
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 64, 64), 3),
        lambda _: nn.Tanh(),
        lambda _: MinMaxPool(lambda: nn.MaxPool2d(2, 2)),  # 16
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 128, 128), 3),
        lambda _: nn.Tanh(),
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 128, 128), 3),
        lambda _: nn.Tanh(),
        lambda _: MinMaxPool(lambda: nn.MaxPool2d(2, 2)),  # 8
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 256, 256), 3),
        lambda _: nn.Tanh(),
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 256, 256), 3),
        lambda _: nn.Tanh(),
        lambda _: MinMaxPool(lambda: nn.MaxPool2d(2, 2)),  # 4
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 512, 512), 3),
        lambda _: nn.Tanh(),
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 512, 512), 3),
        lambda _: nn.Tanh(),
        lambda _: MinMaxPool(lambda: nn.MaxPool2d(2, 2)),  # 2
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 1024, 512), 3),
        lambda _: nn.Tanh(),
        lambda x: FixedPCA(next_pca_weights(x, tn_repr, 3, 512, 512), 3),
        lambda _: nn.Tanh(),
        lambda _: MinMaxPool(lambda: nn.AdaptiveMaxPool2d((1, 1))),  # 1
        lambda _: nn.Flatten(),
    ]
    encoder_layers = []
    encoded_x = train_x
    for layer_config in tqdm.tqdm(encoder_config):
        with torch.no_grad():
            layer = layer_config(encoded_x).to(tn_repr)
            encoder_layers.append(layer)
            encoded_x = [layer(x.to(tn_repr)) for x in encoded_x]
    cla = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 10))
    model = nn.Sequential(*encoder_layers, cla).to(tn_repr)
    opt = torch.optim.Adam(cla.parameters())
    prog = tqdm.trange(50)
    test_x, test_y = list(get_test_cifar(10000))[0]
    t_acc = 0.0
    for epoch in prog:
        for x, y in train:
            x, y = x.to(tn_repr), y.to(tn_repr.device)
            opt.zero_grad()
            rob_loss = (cla[-1].weight.abs().sum(-1) ** 2).sum().sqrt()
            cls_loss = F.cross_entropy(model(x), y)
            (cls_loss + rob_loss * 3e-5).backward()
            prog.set_description("C: %.4f, R: %.4f, T: %.4f" % (cls_loss, rob_loss, t_acc))
            opt.step()
        with torch.no_grad():
            t_acc = (
                WrappedModel(model, 500)(test_x.to(tn_repr)).cpu().numpy().argmax(-1)
                == test_y.numpy()
            ).mean()
    eval_model.eval_model_pgd(model, test, tn_repr.device, 4 / 255, 8 / 255, 20, 'cifar10')
    eval_model.eval_model_with_attack(
        WrappedModel(model, 500), get_test_cifar(10000),
        EnergyAttack(4 / 255, 8 / 255, 10000),
        8 / 255, tn_repr.device, 'cifar10'
    )
