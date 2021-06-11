import numpy
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
        import torch.jit
        tmod = torch.jit.trace(model, x)
        del tmod.training
        tmod.eval()
        tmod = torch.jit.freeze(tmod)
        code, consts = tmod.code_with_constants
        print()
        print(code)
        print(consts.const_mapping.keys())
        # print(consts.c123)

        with torch.no_grad():
            def fn(cx): return torch.trace(model(cx)[..., y])
            fn_orig = fn(x)
            print("O", fn_orig)
            tries = []
            for _ in range(50):
                ap = torch.sign(torch.randn_like(x)) * self.epsilon
                d1 = fn(x + ap * 0.5) - fn_orig
                d2 = fn(x + ap) - fn_orig
                # print("D1", d1)
                # print("D2", d2)
                tries.append(abs(d2 / (2 * d1)).item())
            print("Max", numpy.max(tries), "Min", numpy.min(
                tries), "Mean", numpy.mean(tries))

        def minimized(cz):
            return torch.trace(F.softmax(model(cz), -1)[..., y])

        def fun(cx):
            with torch.no_grad():
                return minimized(
                    x.new_tensor(cx.reshape(*x.shape))).double().item()

        def jac(cx):
            return torch.autograd.functional.jacobian(
                minimized,
                x.new_tensor(cx.reshape(*x.shape))
            ).cpu().double().numpy().reshape(-1)

        def clbk(_):
            return False

        result = opt.minimize(
            fun, (x + torch.sign(torch.randn_like(x)) *
                  self.epsilon).cpu().numpy().reshape(-1),
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

        return x_adv
