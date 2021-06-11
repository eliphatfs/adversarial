import torch
import torch.nn.functional as F


class SPSA():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps * 2
        self.random_start = random_start
        self.step_size = epsilon / 20

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        succeeded_attacks = x.detach()
        with torch.no_grad():
            for i in range(self.perturb_steps):
                pertub = torch.sign(torch.randn_like(x_adv))

                x_plus = x_adv.detach() + pertub.detach()*self.step_size
                x_plus = torch.min(
                    torch.max(x_plus, x - self.epsilon), x + self.epsilon)
                x_plus = torch.clamp(x_plus, 0.0, 1.0)
                y_plus = F.cross_entropy(model(x_plus), y)

                x_minus = x_adv.detach() - pertub.detach()*self.step_size
                x_minus = torch.min(
                    torch.max(x_minus, x - self.epsilon), x + self.epsilon)
                x_minus = torch.clamp(x_minus, 0.0, 1.0)
                y_minus = F.cross_entropy(model(x_minus), y)

                gp = (y_plus - y_minus) / (2*self.step_size)
                x_adv = x_adv.detach() + torch.sign(gp * pertub)
                x_adv = torch.min(
                    torch.max(x_adv, x - self.epsilon), x + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

                for idx, (img, label) in enumerate(zip(x_adv, y)):
                    if model(img.unsqueeze(0)).max(-1)[-1].item() != label.item():
                        succeeded_attacks[idx] = img

        return succeeded_attacks
