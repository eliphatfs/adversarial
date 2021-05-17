import torch
import torch.nn.functional as F
import scipy.optimize as opt
import numpy


class ExpAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.processed_model_id = None

    def bad_krylov(self, model, x, y):
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

    def __call__(self, model, x, y):
        return self.frank_wolfe(model, x, y)
        model.eval()

        def func(cx):
            logits = model(cx)
            # labels = torch.eye(10).to(y.device)[y]
            return torch.nn.functional.cross_entropy(logits, y)

        def safe_jac(cx):
            jac = torch.autograd.functional.jacobian(func, cx)
            jac = torch.randn_like(jac) * 1e-7 + jac
            return torch.sign(jac)

        krylov = [safe_jac(x)]
        b = krylov[0]
        for i in range(self.perturb_steps):
            krylov.append(safe_jac(x + krylov[i] * self.epsilon))
        U = torch.stack(krylov[:-1])
        AU = torch.stack(krylov[1:])
        x_adv = x.detach().clone()
        delta = b * self.epsilon
        lam = torch.norm(b.flatten(1), 2, dim=-1) / b.flatten(1).shape[-1] ** 0.5
        for _ in range(4):
            A = (AU.flatten(2) - lam.reshape(1, -1, 1) * U.flatten(2)).permute(1, 2, 0)  # [n, *, m]
            B = -b.flatten(1).unsqueeze(-1)  # [n, *, 1]
            # AtA k = At B
            k, _ = torch.solve(A.transpose(2, 1).bmm(B), A.transpose(2, 1).bmm(A))
            Uf = U.flatten(2).permute(1, 2, 0)  # [n, *, m]
            delta = Uf.bmm(k).reshape(*x.shape)
            lam = torch.norm(AU.flatten(2).permute(1, 2, 0).bmm(k).squeeze(-1), 2, dim=-1) / b.flatten(1).shape[-1] ** 0.5
            with torch.no_grad():
                pertubed_sample = torch.min(
                    torch.max(x + delta, x - self.epsilon), x + self.epsilon
                )
                pertubed_sample = torch.clamp(pertubed_sample, 0.0, 1.0)
                y_pred = model(pertubed_sample).argmax(-1)
                for i, (lab_pred, lab_true) in enumerate(zip(y_pred, y)):
                    if lab_pred != lab_true:
                        x_adv[i] = pertubed_sample[i]
                pertubed_sample = x + torch.sign(delta) * self.epsilon
                pertubed_sample = torch.clamp(pertubed_sample, 0.0, 1.0)
                y_pred = model(pertubed_sample).argmax(-1)
                for i, (lab_pred, lab_true) in enumerate(zip(y_pred, y)):
                    if lab_pred != lab_true:
                        x_adv[i] = pertubed_sample[i]
        return x_adv


    def frank_wolfe(self, model, x, y):
        model.eval()
        
        def func(cx):
            logits = model(cx)
            # labels = torch.eye(10).to(y.device)[y]
            return torch.nn.functional.cross_entropy(logits, y)

        x_adv = x.detach()
        for k in range(self.perturb_steps):
            jac = torch.autograd.functional.jacobian(func, x_adv)
            s = x + torch.sign(jac) * self.epsilon
            a = 2 / (k + 2)
            x_adv = x_adv + a * (s - x_adv)
        return x_adv
