import torch
import torch.nn.functional as F
import scipy.optimize as opt


class BetterSecondOrderAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        minimized = lambda cz: -F.cross_entropy(model(cz), y)
    
        def fun(cx):
            with torch.no_grad():
                return minimized(x.new_tensor(cx.reshape(*x.shape))).double().item()

        def jac(cx):
            return torch.autograd.functional.jacobian(
                minimized,
                x.new_tensor(cx.reshape(*x.shape))
            ).cpu().double().numpy().reshape(-1)

        def clbk(_):
            return False

        result = opt.minimize(
            fun, x.cpu().numpy().reshape(-1),
            method="L-BFGS-B", jac=jac,
            bounds=[
                (max(0, v - self.epsilon), min(1, v + self.epsilon))
                for v in x.cpu().double().numpy().reshape(-1)
            ],
            callback=clbk,
            options={
                'maxiter': self.perturb_steps * 5,
                'maxcor': 100,
                # 'minfev': -6,
                # 'disp': True,
                # 'maxfun': 300,
                'iprint': 1
            }
        )
        x_adv = torch.min(
            torch.max(
                x.new_tensor(result.x.reshape(*x.shape)),
                x - self.epsilon
            ),
            x + self.epsilon
        )
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv
