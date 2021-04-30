import torch
import torch.nn.functional as F


class HybridAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        succeeded_attacks = x.detach()
        sobol_eng = torch.quasirandom.SobolEngine(x_adv.flatten(0).shape[0])
        for i in range(10):
            pertub = torch.sign(sobol_eng.draw(1).to(x_adv.device)-0.5).reshape_as(x_adv)
            x_adv = x.detach() + pertub.detach()
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            for idx, (batch, batch_label, batch_grad) in enumerate(zip(x_adv, y, grad)):
                if torch.linalg.norm(batch_grad.flatten(0)) > 1e-7:
                    # non-zero grad, use pgd
                    for i in range(self.perturb_steps):
                        batch.requires_grad_()
                        with torch.enable_grad():
                            loss_c = F.cross_entropy(model(batch.unsqueeze(0)), batch_label.unsqueeze(0))
                        grad = torch.autograd.grad(loss_c, [batch])[0]
                        batch = batch.detach() + self.step_size * torch.sign(grad.detach())
                        batch = torch.min(
                            torch.max(batch, x[idx] - self.epsilon), x[idx] + self.epsilon)
                        batch = torch.clamp(batch, 0.0, 1.0)
                    if model(batch.unsqueeze(0)).max(-1)[-1].item() != batch_label.item():
                        succeeded_attacks[idx] = batch
        return succeeded_attacks
