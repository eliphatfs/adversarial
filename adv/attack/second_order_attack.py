import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class P(nn.Module):
    def __init__(self, x_adv):
        super().__init__()
        self.params = nn.Parameter(x_adv)


class SecondOrderAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        P_module = P(torch.randn_like(x_adv))
        optimizer = torch.optim.LBFGS(
            P_module.parameters(), line_search_fn='strong_wolfe')
        pertub = P_module.params

        def closure():
            optimizer.zero_grad()
            # params_clamped = torch.min(
            #     torch.max(params, x - self.epsilon), x + self.epsilon)
            # params_clamped = torch.clamp(params_clamped, 0.0, 1.0)
            with torch.enable_grad():
                loss_params = - \
                    F.cross_entropy(model(pertub+x.detach()), y) + \
                    (pertub**2).sum()**0.5 * 0.1
                # loss_params = - F.mse_loss(model(params_clamped), pred)
            loss_params.backward()

            return loss_params

        for _ in range(self.perturb_steps):
            print("Before Stepping: {:.4f}".format(abs(x_adv-x).max().item()))
            optimizer.step(closure)
            print("Unclamped: {:.4f}".format(abs(pertub).max().item()))
            x_adv = torch.min(
                torch.max(pertub+x, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            print("Clamped: {:.4f}".format(abs(x_adv - x).max().item()))
            print()
        return x_adv
