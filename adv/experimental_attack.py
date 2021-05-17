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
        model.eval()

        def func(cx):
            logits = model(cx)
            return torch.nn.functional.cross_entropy(logits, y)

        def safe_jac(cx):
            jac = torch.autograd.functional.jacobian(func, cx)
            jac = torch.randn_like(jac) * 1e-7 + jac
            return jac

        tg = x.flatten(1).shape[-1] ** 0.5
        b = safe_jac(x)
        krylov = [torch.sign(b) / tg]
        batch_dot = lambda u, v: torch.bmm(u.flatten(1).unsqueeze(-2), v.flatten(1).unsqueeze(-1)).flatten(0)
        H = torch.zeros(
            [x.shape[0], self.perturb_steps + 1, self.perturb_steps],
            dtype=x.dtype,
            device=x.device
        )
        last = krylov[-1]
        for i in range(self.perturb_steps):
            v = safe_jac(x + last * self.epsilon * 0.7) - b
            for j, u in enumerate(krylov):
                H[:, j, i] = batch_dot(v, u)
                v = v - H[:, j, i].reshape(-1, 1, 1, 1) * u
            H[:, i + 1, i] = torch.linalg.norm(v.flatten(1), dim=-1)
            v = v / H[:, i + 1, i].reshape(-1, 1, 1, 1)
            krylov.append(v)
            last = v
        U = torch.stack(krylov[:-1])
        Uf = U.flatten(2).permute(1, 2, 0)  # n, *, m
        # x_adv = (x + torch.sign(b) * self.epsilon).detach().clone()
        x_adv = x.detach().clone()


    def not_so_bad_krylov(self, model, x, y):
        model.eval()

        def func(cx):
            logits = model(cx)
            # labels = torch.eye(10).to(y.device)[y]
            return torch.nn.functional.cross_entropy(logits, y)

        def safe_jac(cx):
            jac = torch.autograd.functional.jacobian(func, cx)
            jac = torch.randn_like(jac) * 1e-7 + jac
            return jac

        tg = x.flatten(1).shape[-1] ** 0.5
        b = safe_jac(x)
        krylov = [b / torch.norm(b.flatten(1), -1).reshape(-1, 1, 1, 1)]
        batch_dot = lambda u, v: torch.bmm(u.flatten(1).unsqueeze(-2), v.flatten(1).unsqueeze(-1)).flatten(0)
        H = torch.zeros(
            [x.shape[0], self.perturb_steps + 1, self.perturb_steps],
            dtype=x.dtype,
            device=x.device
        )
        last = krylov[-1]
        for i in range(self.perturb_steps):
            v = safe_jac(x + last * self.epsilon * tg) - b
            for j, u in enumerate(krylov):
                H[:, j, i] = batch_dot(v, u)
                v = v - H[:, j, i].reshape(-1, 1, 1, 1) * u
            H[:, i + 1, i] = torch.linalg.norm(v.flatten(1), dim=-1)
            v = v / H[:, i + 1, i].reshape(-1, 1, 1, 1)
            krylov.append(v)
            last = v
        U = torch.stack(krylov[:-1])
        # AU = torch.stack(krylov[1:])
        # x_adv = (x + torch.sign(b) * self.epsilon).detach().clone()
        x_adv = x.detach().clone()
        # delta = b * self.epsilon
        # .reshape(1, -1, 1)

        def resolve(lam):
            # A = (AU.flatten(2) - lam.reshape(1, -1, 1) * U.flatten(2)).permute(1, 2, 0)  # [n, *, m]
            # AtA k = At B
            # k, _ = torch.solve(A.transpose(2, 1).bmm(B), A.transpose(2, 1).bmm(A))
            Uf = U.flatten(2).permute(1, 2, 0)  # [n, *, m]
            B = b.flatten(1).unsqueeze(-1)  # [n, *, 1]
            # (H - lam I) k = -Ut B
            k, _ = torch.solve(
                -Uf.transpose(2, 1).bmm(B),
                H[:, :-1, :] - lam * Uf.transpose(2, 1).bmm(Uf)
            )
            delta = Uf.bmm(k).reshape(*x.shape)
            return delta

        # lam = torch.norm(b.flatten(1), 2, dim=-1) / b.flatten(1).shape[-1] ** 0.5
        # print(lam)
        lam = 5
        active_indices = list(range(x.shape[0]))
        removed = set()
        for _ in range(19):
            delta = resolve(lam)
            fooled = 0
            with torch.no_grad():
                perturbed_sample = x + torch.sign(delta) * self.epsilon
                perturbed_sample = torch.clamp(perturbed_sample, 0.0, 1.0)
                y_pred = model(perturbed_sample[active_indices]).argmax(-1)
                for i, lab_pred, lab_true in zip(active_indices, y_pred, y[active_indices]):
                    if lab_pred != lab_true:
                        x_adv[i] = perturbed_sample[i]
                        removed.add(i)
                        fooled += 1
                active_indices = [i for i in active_indices if i not in removed]
            # print(lam, fooled)
            lam *= 3 / 4
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
