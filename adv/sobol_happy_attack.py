import torch
import torch.nn.functional as F


class SobolHappyAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps * 10
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        succeeded_attacks = x.detach().clone()
        sobol_eng = torch.quasirandom.SobolEngine(x_adv.flatten(1).shape[-1])
        for i in range(self.perturb_steps):
            pertub = torch.sign(sobol_eng.draw(1).to(x_adv.device)-0.5).reshape(*x_adv.shape[1:])
            x_adv = x.detach() + pertub.detach()
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            with torch.no_grad():
                y_pred = model(x_adv).argmax(-1)
            for i, (lab_pred, lab_true) in enumerate(zip(y_pred, y)):
                if lab_pred != lab_true:
                    succeeded_attacks[i] = x_adv[i]
        return succeeded_attacks
