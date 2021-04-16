import torch
import torch.nn.functional as F


class PNewtonAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            hessian = torch.autograd.functional.hessian(
                lambda x_adv: F.cross_entropy(model(x_adv.reshape(-1, 3, 32, 32)), y),
                x_adv.flatten(0))
            descent_direction = torch.linalg.pinv(hessian).detach().mm(grad.reshape(-1, 1)).reshape(-1, 3, 32, 32)
            print(grad.max())
            print(torch.linalg.norm(grad.flatten(0)))
            x_adv = x_adv.detach()\
                + self.step_size * descent_direction
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv
