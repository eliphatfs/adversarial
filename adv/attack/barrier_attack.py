import torch
import torch.nn as nn
import torch.optim as optim


class P(nn.Module):
    def __init__(self, x_adv):
        super().__init__()
        self.params = nn.Parameter(x_adv)


class BarrierMethodAttack:
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        adv = P(x.detach().clone()).to(x.device)
        opt = optim.LBFGS(adv.parameters(), line_search_fn='strong_wolfe')
        t = 1

        def closure():
            opt.zero_grad()
            x_adv = adv.params
            inf_norm = torch.max(torch.abs(x_adv - x))
            # inf_norm - eps <= 0
            f = nn.functional.cross_entropy(model(x_adv), y)
            loss = - torch.log(self.epsilon - inf_norm) / t - f
            loss.backward()
            return loss

        for outer in range(3):
            for inner in range(4):
                opt.step(closure)
            t *= 10

        return adv.params
