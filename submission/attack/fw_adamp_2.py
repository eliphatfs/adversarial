import torch
import torch.nn.functional as F


class FWAdampAttackPlus():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        return self.frank_wolfe(model, x, y)

    def safe_jac(self, model, x, y):
        x_rg = x.detach().clone().requires_grad_(True)
        logits = model(x_rg)
        L = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # AdAmp (maybe not a good name for the op)
        adamp_loss = sum(w ** 2 * F.cross_entropy(logits * w, y) for w in L)
        jac = torch.autograd.grad(adamp_loss, [x_rg])[0]
        jac_norm = torch.norm(jac.flatten(1), dim=-1).reshape(-1, 1, 1, 1)
        return jac / (jac_norm + 1e-7) + torch.randn_like(jac) * 5e-8

    def frank_wolfe(self, model, x, y):
        model.eval()

        x_adv = x.detach()
        for k in range(self.perturb_steps):
            safe_jac = self.safe_jac(model, x_adv, y)
            s = x + torch.sign(safe_jac) * self.epsilon
            a = 2 / (k + 2)
            x_adv = x_adv + a * (s - x_adv)
        return torch.clamp(x_adv, 0, 1)

    def targeted(self, model, x, y_tgt):
        model.eval()

        x_adv = x.detach()
        for k in range(self.perturb_steps):
            safe_jac = self.safe_jac(model, x_adv, y_tgt)
            s = x - torch.sign(safe_jac) * self.epsilon
            a = 2 / (k + 2)
            x_adv = x_adv + a * (s - x_adv)
        return torch.clamp(x_adv, 0, 1)
