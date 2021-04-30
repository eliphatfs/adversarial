import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class P(nn.Module):
    def __init__(self, x_adv):
        super().__init__()
        self.params = nn.Parameter(x_adv)


class DeepFoolAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        # x_adv.requires_grad_()
        for i in range(self.perturb_steps):
            for idx, (batch_x, batch_y) in enumerate(zip(x_adv, y)):
                batch_x = torch.tensor(batch_x, requires_grad=True)
                l_min = float('inf')
                with torch.enable_grad():
                    pred = model(batch_x.unsqueeze(0))[0]
                batch_grad = torch.autograd.grad(pred[batch_y], [batch_x], retain_graph=True)[0]
                total_pertub = torch.zeros_like(batch_x)
                for k, val in enumerate(pred):
                    if k == batch_y:
                        continue
                    w = torch.autograd.grad(pred[k], [batch_x], retain_graph=True)[0] - batch_grad
                    f = val - pred[batch_y]
                    L = torch.linalg.norm(f.flatten(0), 1) / torch.linalg.norm(w.flatten(0), 1)
                    if L < l_min:
                        l_min = L
                        w_l = w
                        f_l = f
                pertub = torch.linalg.norm(f_l.flatten(0), 1) / torch.linalg.norm(w_l.flatten(0), 1) * torch.sign(w_l)
                x_adv[idx] = batch_x + pertub
                total_pertub = total_pertub + pertub

        print("Unclamped: {:.4f}".format(abs(pertub).max().item()))
        x_adv = torch.min(
            torch.max(pertub+x, x - self.epsilon), x + self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        print("Clamped: {:.4f}".format(abs(x_adv - x).max().item()))
        print()
        return x_adv
