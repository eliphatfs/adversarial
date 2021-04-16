import torch
import torch.nn.functional as F


class PGDAttackMod2():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        for i in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # value, index = model(x_adv).max(-1)
                # loss_c = value.sum()
                loss_c = 1000 * F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            print(grad.max())
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            if i == 0 and grad.max().item() == 0:
                print('Failed.')
        return x_adv
