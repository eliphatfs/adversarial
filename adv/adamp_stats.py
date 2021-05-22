import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange


def adamp_counter(model: nn.Module, x_adv: torch.Tensor, y: torch.Tensor):
    L = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    adamp_pure = []
    adamp_lost = []
    ce_pure = []
    ce_lost = []
    batch, _, _, _ = x_adv.shape
    with torch.no_grad():
        logits = model(x_adv)  # F.softmax(model(cx), -1) (batch, num_classes)
        for b in range(batch):
            pred = logits[b, :].unsqueeze(0)
            label = y[b].unsqueeze(0)
            ce = F.cross_entropy(pred, label)
            adamp = sum(w ** 2 * F.cross_entropy(pred * w, label) for w in L)
            if pred.argmax(dim=1) == y[b]:
                ce_pure.append(ce.item())
                adamp_pure.append(adamp.item())
            else:
                ce_lost.append(ce.item())
                adamp_lost.append(adamp.item())
        return ce_pure, ce_lost, adamp_pure, adamp_lost


def attack_and_analyze_stats(model, test_loader, attack, epsilon, device):
    correct_adv, correct = [], []
    ce_pure, ce_lost, adamp_pure, adamp_lost = [], [], [], []
    distance = []
    num = 0
    with trange(10000) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, _, _, _ = x.shape
            x_adv = attack(model, x.clone(), label.clone())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = x_adv.clamp(0,1)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
                cp, cl, ap, al = adamp_counter(model, x_adv, label)
                ce_pure += cp
                ce_lost += cl
                adamp_pure += ap
                adamp_lost += al
            distance.append(torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            num += x.shape[0]
            pbar.set_description(f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    distance = torch.cat(distance).max()
    return ce_pure, ce_lost, adamp_pure, adamp_lost
