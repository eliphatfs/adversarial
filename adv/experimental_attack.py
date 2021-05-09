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
        y_cpu = y.cpu().numpy()
        results = x.detach().clone()
        krylov = []
        # batch_dot = lambda u, v: torch.bmm(u.unsqueeze(-2), v.unsqueeze(-1)).squeeze(-1)
        for _ in range(6):
            jac = torch.autograd.functional.jacobian(func, x_adv)
            x_adv = x + torch.sign(jac) * self.epsilon
            krylov.append(torch.sign(jac))
            '''v = (torch.sign(jac) + torch.randn_like(jac) * 1e-5).flatten(1)
            for u in krylov:
                v = v - batch_dot(v, u) * u
            krylov.append(v / torch.linalg.norm(v, 2, -1, keepdim=True))'''
        fix, krylov = krylov[0], krylov[1:]
        for i in range(2 ** 5):
            q = torch.zeros_like(x) + fix
            for kv in range(5):
                v = krylov[kv] if (i & (1 << kv)) > 0 else -krylov[kv]
                q = q + v.reshape(*x.shape)
            x_adv = x + torch.sign(q) * self.epsilon
            pred = model(x_adv).argmax(-1).cpu().numpy()
            for i in range(len(pred)):
                if pred[i] != y_cpu[i]:
                    results[i] = x_adv[i]
        return results
        '''x_adv = x.detach()
        for k in range(self.perturb_steps):
            jac = torch.autograd.functional.jacobian(func, x_adv)
            s = x + torch.sign(jac) * self.epsilon
            a = 2 / (k + 2)
            x_adv = x_adv + a * (s - x_adv)
        return x_adv'''
