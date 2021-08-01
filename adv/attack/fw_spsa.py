import torch
import torch.nn.functional as F


class StochasticFWAdampAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps * 2
        self.random_start = random_start
        self.step_size = epsilon / 20

    def __call__(self, model, x, y):
        return self.fw_spsa(model, x, y)

    def fw_spsa(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        succeeded_attacks = x.detach()
        with torch.no_grad():
            for k in range(self.perturb_steps):
                perturb = torch.sign(torch.randn_like(x_adv))

                x_plus = x_adv.detach() + perturb.detach()*self.step_size
                x_plus = torch.min(
                    torch.max(x_plus, x - self.epsilon), x + self.epsilon)
                x_plus = torch.clamp(x_plus, 0.0, 1.0)
                y_plus = F.cross_entropy(model(x_plus), y)

                x_minus = x_adv.detach() - perturb.detach()*self.step_size
                x_minus = torch.min(
                    torch.max(x_minus, x - self.epsilon), x + self.epsilon)
                x_minus = torch.clamp(x_minus, 0.0, 1.0)
                y_minus = F.cross_entropy(model(x_minus), y)

                gp = (y_plus - y_minus) / (2*self.step_size)

                s = x + torch.sign(gp * perturb)
                a = 2 / (k + 2)
                x_adv = x_adv + a * (s - x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)

                y_pred = model(x_adv).argmax(-1)
                for idx, (lab_pred, lab_true) in enumerate(zip(y_pred, y)):
                    if lab_pred != lab_true:
                        succeeded_attacks[idx] = x_adv[idx]

        return succeeded_attacks
