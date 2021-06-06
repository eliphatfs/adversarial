import torch
import torch.nn.functional as F


class ChihaoHappyAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps * 100
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        succeeded_attacks = x.detach().clone()
        for i in range(self.perturb_steps):
            pertub = torch.sign(torch.randn_like(x_adv))
            x_adv = x_adv.detach() + pertub.detach()
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            for idx, (img, label) in enumerate(zip(x_adv, y)):
                if model(img.unsqueeze(0)).max(-1)[-1].item() != label.item():
                    succeeded_attacks[idx] = img
        return succeeded_attacks
