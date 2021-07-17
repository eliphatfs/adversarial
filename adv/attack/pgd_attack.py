import torch
import torch.nn.functional as F


def pgd_attack(model, x, y, step_size, epsilon, perturb_steps,
               random_start=None, distance='l_inf'):
    model.eval()
    x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


class PGDAttack():
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
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def untargeted(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        for i in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def targeted(self, model, x, y_tgt):
        model.eval()
        x_adv = x.detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y_tgt)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() - self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv
