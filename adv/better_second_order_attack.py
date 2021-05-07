import random
import numpy
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

        with torch.no_grad():
            fn = lambda cx: torch.trace(model(cx)[..., y])
            fn_orig = fn(x)
            print("O", fn_orig)
            tries = []
            for _ in range(50):
                ap = torch.sign(torch.randn_like(x)) * self.epsilon
                d1 = fn(x + ap * 0.5) - fn_orig
                d2 = fn(x + ap) - fn_orig
                # print("D1", d1)
                # print("D2", d2)
                tries.append(abs(d2 - 2 * d1).item())
            print("Max", numpy.max(tries), "Min", numpy.min(tries), "Mean", numpy.mean(tries))

        def minimized(cz):
            return torch.trace(F.softmax(model(cz), -1)[..., y])
            return -F.l1_loss(F.softmax(model(cz), -1), F.softmax(model(x), -1))
            logits = model(cz)
            prob = F.softmax(logits, -1)
            s = torch.argsort(prob, -1, descending=True)
            targets = (
                (s[..., 0] != y).long() * s[..., 0]
                + (s[..., 1] != y).long() * s[..., 1]
            )
            return F.cross_entropy(logits, targets)
            # exp_mat = torch.sigmoid(model(cz))
            # return torch.trace(exp_mat[..., y]) / torch.sum(exp_mat)

        '''with torch.no_grad():
            torch.set_printoptions(precision=3, sci_mode=False)
            print(F.softmax(model(x), -1))
            print(torch.diagonal(F.softmax(model(x), -1)[..., y]))'''
    
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
        
        # torch.sign(torch.randn_like(x)) * self.epsilon
        '''return torch.sign(-x.new_tensor(jac(
            (x + torch.sign(torch.randn_like(x)) * self.epsilon).cpu().numpy().reshape(-1)
        )).reshape(*x.shape)) * self.epsilon + x'''

        result = opt.minimize(
            fun, (x + torch.sign(torch.randn_like(x)) * self.epsilon).cpu().numpy().reshape(-1),
            method="L-BFGS-B", jac=jac,
            bounds=[
                (max(0, v - self.epsilon), min(1, v + self.epsilon))
                for v in x.cpu().double().numpy().reshape(-1)
            ],
            callback=clbk,
            options={
                'maxiter': self.perturb_steps,
                'maxcor': 100,
                # 'minfev': -6,
                # 'disp': True,
                # 'maxfun': 300,
                # 'iprint': 1
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
        '''with torch.no_grad():
            print(F.softmax(model(x_adv), -1))
            print(torch.diagonal(F.softmax(model(x_adv), -1)[..., y]))'''
        return x_adv
