import torch


class ExpAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.processed_model_id = None

    def __call__(self, model, x, y):
        model.eval()
        
        def func(cx):
            logits = model(cx)
            # labels = torch.eye(10).to(y.device)[y]
            return torch.nn.functional.cross_entropy(logits, y)
            diag = torch.diagonal(logits[..., y])
            return logits.sum() - 2 * diag.sum()

        x_adv = x.detach()
        for k in range(self.perturb_steps):
            jac = torch.autograd.functional.jacobian(func, x_adv)
            s = x + torch.sign(jac) * self.epsilon
            a = 2 / (k + 2)
            x_adv = x_adv + a * (s - x_adv)
        return x_adv
