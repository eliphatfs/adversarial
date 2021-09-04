from utils import prepare_cifar, get_test_cifar
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import eval_model
import math
from attack.energy_attack import EnergyAttack


def patches_of(bchw, kernel=5, stride=1, pad_same=True, zero_mean=True):
    b, c, h, w = bchw.shape
    if pad_same:
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_h = (h2 - 1) * stride + (kernel - 1) + 1 - h
        pad_w = (w2 - 1) * stride + (kernel - 1) + 1 - w
        x = F.pad(bchw, (pad_h//2, pad_h - pad_h//2, pad_w//2, pad_w - pad_w//2))
    else:
        x = bchw
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()
    
    patches = patches.view(b, -1, patches.shape[-2], patches.shape[-1])
    if zero_mean:
        return patches - patches.mean([-1, -2, -3], keepdim=True)
    else:
        return patches


class MinMaxPool(nn.Module):
    def __init__(self, maxpool_gen):
        super().__init__()
        self.mp1 = maxpool_gen()
        self.mp2 = maxpool_gen()

    def forward(self, x):
        return torch.cat([self.mp1(x), self.mp2(-x)], 1)


class FixedPCA(nn.Module):
    def __init__(self, coef, kernel):
        super().__init__()
        self.coef = nn.Parameter(coef, False)
        self.kernel = kernel

    def forward(self, x):
        patch = patches_of(x, self.kernel)
        r = torch.einsum("bihw,ij->bjhw", patch, self.coef)
        return torch.cat([r, -r], 1)


def next_pca_weights(image_loader, tn_repr, kernel, in_feat, out_feat):
    cov = torch.zeros([kernel ** 2 * in_feat] * 2).to(tn_repr)
    for x in image_loader:
        patch = patches_of(x.to(tn_repr), kernel, pad_same=False)
        cov = cov + torch.einsum('bphw,bqhw->pq', patch, patch)
    sigma, eigv = torch.linalg.eigh(cov)
    return eigv[:, -out_feat:]


def next_lda_weights(image_loader, tn_repr, kernel, in_feat, out_feat):
    mu = torch.zeros([kernel ** 2 * in_feat]).to(tn_repr)
    w = 0
    for x in image_loader:
        patch = patches_of(x.to(tn_repr), kernel, zero_mean=False)
        mu = mu + patch.mean([-1, -2]).sum(0)
        w += len(patch)
    sw = torch.zeros([kernel ** 2 * in_feat] * 2).to(tn_repr)
    st = torch.zeros([kernel ** 2 * in_feat] * 2).to(tn_repr)
    mu = mu / w
    for x in image_loader:
        patch = patches_of(x.to(tn_repr), kernel, zero_mean=False)
        internal = patch - patch.mean([-1, -2], keepdim=True)
        total = patch - mu.view(-1, 1, 1)
        sw = sw + torch.einsum("bphw,bqhw->pq", internal, internal)
        st = st + torch.einsum("bphw,bqhw->pq", total, total)
    sj = torch.linalg.solve(sw, st - sw)
    sigma, eigv = torch.linalg.eigh(sj)
    return eigv[:, -out_feat:]


def next_cpt_weights(image_loader, tn_repr, kernel, in_feat, out_feat):
    c = 0
    for x in image_loader:
        patch = patches_of(x.to(tn_repr), kernel, zero_mean=False)
        c += patch[:, 0].numel()

    w = torch.randn([c, out_feat]).to(tn_repr).abs() * math.sqrt(0.5 / out_feat)
    h = torch.randn([in_feat * kernel ** 2, out_feat]).to(tn_repr).abs() * math.sqrt(0.5 / out_feat)

    def update_rule():
        loss = 0.0
        rh = torch.zeros_like(h)
        rw = torch.zeros_like(w)
        c = 0
        for x in image_loader:
            patch = patches_of(x.to(tn_repr), kernel, zero_mean=False)
            patch = patch.permute(1, 0, 2, 3).flatten(1)  # [f, L]
            L = patch[0].numel()
            rh += patch.mm(w[c: c + L])  # [f, L], [L, o] -> [f, o]
            rw[c: c + L] = patch.t().mm(h)  # [L, f], [f, o] -> [L, o]
            loss += ((patch.t() - w[c: c + L].mm(h.t())) ** 2).sum()
            c += L
        rh *= h
        rh /= h.mm(w.t().mm(w))
        rw *= w
        rw /= w.mm(h.t().mm(h))
        rh.nan_to_num_(0, 0, 0)
        rw.nan_to_num_(0, 0, 0)
        return loss, rw, rh

    while True:
        loss, rw, rh = update_rule()
        print(loss)
        w = rw
        loss, rw, rh = update_rule()
        print(loss, (rh - h).abs().sum())
        if (rh - h).abs().sum() < 0.1:
            return rh
        h = rh


class WrappedModel(nn.Module):
    def __init__(self, wrap, subbatch):
        super().__init__()
        self.wrap = wrap
        self.subbatch = subbatch

    def forward(self, x):
        subbatches = torch.split(x, self.subbatch)
        return torch.cat([self.wrap(sb) for sb in subbatches])


class SomeActivation(nn.Module):
    def forward(self, x):
        return F.relu(x)


class InputProcess(nn.Module):
    def forward(self, x):
        return (x - 0.5) / 0.5


if __name__ == '__main__':
    train, test = prepare_cifar(400, 400)
    tn_repr = torch.zeros([3]).to('cuda:1')
    train_x = [x for x, _ in train]
    encoder_config = [
        lambda x: FixedPCA(next_cpt_weights(x, tn_repr, 5, 3, 32), 5),
        lambda _: nn.BatchNorm2d(64, affine=False, track_running_stats=False),
        lambda _: SomeActivation(),
        lambda _: nn.MaxPool2d(2, 2),  # 16
        # lambda x: FixedPCA(next_cpt_weights(x, tn_repr, 3, 64, 64), 3),
        # lambda _: nn.BatchNorm2d(128, affine=False, track_running_stats=False),
        # lambda _: SomeActivation(),
        # lambda _: nn.MaxPool2d(2, 2),  # 8
        # lambda x: FixedPCA(next_cpt_weights(x, tn_repr, 3, 128, 128), 3),
        # lambda _: nn.BatchNorm2d(256, affine=False, track_running_stats=False),
        # lambda _: SomeActivation(),
        # # lambda x: FixedPCA(next_lda_weights(x, tn_repr, 3, 256, 128), 3),
        # # lambda _: nn.BatchNorm2d(256, affine=False, track_running_stats=False),
        # # lambda _: SomeActivation(),
        # lambda _: nn.MaxPool2d(2, 2),  # 4
        # lambda x: FixedPCA(next_cpt_weights(x, tn_repr, 3, 256, 256), 3),
        # lambda _: nn.BatchNorm2d(512, affine=False, track_running_stats=False),
        # lambda _: SomeActivation(),
        # lambda x: FixedPCA(next_lda_weights(x, tn_repr, 3, 512, 256), 3),
        # lambda _: nn.BatchNorm2d(512, affine=False, track_running_stats=False),
        # lambda _: SomeActivation(),
        # lambda _: nn.MaxPool2d(2, 2),  # 2
        # lambda x: FixedPCA(next_lda_weights(x, tn_repr, 3, 512, 512), 3),
        # lambda _: SomeActivation(),
        # lambda _: MinMaxPool(lambda: nn.AdaptiveMaxPool2d((1, 1))),  # 1
        lambda _: nn.Flatten(),
    ]
    encoder_layers = []
    encoded_x = train_x
    for layer_config in tqdm.tqdm(encoder_config):
        with torch.no_grad():
            layer = layer_config(encoded_x).to(tn_repr)
            encoder_layers.append(layer)
            encoded_x = [layer(x.to(tn_repr)) for x in encoded_x]
    cla = nn.Sequential(nn.LazyLinear(10))
    model = nn.Sequential(*encoder_layers, cla).to(tn_repr)
    opt = torch.optim.Adam(cla.parameters())
    prog = tqdm.trange(50)
    test_x, test_y = list(get_test_cifar(10000))[0]
    t_acc = 0.0
    for epoch in prog:
        model.train()
        for x, y in train:
            x, y = x.to(tn_repr), y.to(tn_repr.device)
            opt.zero_grad()
            pred = model(x)
            cls_loss = F.cross_entropy(pred, y)
            rob_loss = (cla[-1].weight.abs().sum(-1) ** 2).sum().sqrt()
            (cls_loss + rob_loss * 0.).backward()
            acc = (pred.max(-1)[-1] == y).float().mean().item()
            prog.set_description("C: %.3f, R: %.3f, A: %.3f, T: %.3f" % (cls_loss, rob_loss, acc, t_acc))
            opt.step()
        with torch.no_grad():
            model.eval()
            t_acc = (
                WrappedModel(model, 400)(test_x.to(tn_repr)).cpu().numpy().argmax(-1)
                == test_y.numpy()
            ).mean()
    eval_model.eval_model_pgd(model, test, tn_repr.device, 4 / 255, 8 / 255, 20, 'cifar10')
    eval_model.eval_model_with_attack(
        WrappedModel(model, 400), get_test_cifar(10000),
        EnergyAttack(4 / 255, 8 / 255, 10000),
        8 / 255, tn_repr.device, 'cifar10'
    )
