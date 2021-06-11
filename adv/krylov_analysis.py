import torch
from utils import get_test_cifar
import matplotlib.pyplot as plotlib
from model import get_model_for_attack
# import vgg

plotlib.style.use('seaborn')
plotlib.rcParams['ps.useafm'] = True
fsize = 24
tsize = 28
parameters = {'axes.labelsize': tsize, 'axes.titlesize': tsize,
              'xtick.labelsize': fsize, 'ytick.labelsize': fsize,
              'legend.fontsize': fsize}
plotlib.rcParams.update(parameters)


test_loader = get_test_cifar(1)
model = get_model_for_attack('model2')
'''model = vgg.vgg13_bn()
ensemble = torch.load('vgg13bn_regm2.dat', map_location=torch.device('cpu'))
model.load_state_dict(ensemble)'''
model.eval()
for x, y in test_loader:
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
    krylov = [b / torch.norm(b.flatten(1), dim=-1).reshape(-1, 1, 1, 1)]

    def batch_dot(u, v): return torch.bmm(
        u.flatten(1).unsqueeze(-2), v.flatten(1).unsqueeze(-1)).flatten(0)
    H = torch.zeros(
        [x.shape[0], 5, 4],
        dtype=x.dtype,
        device=x.device
    )
    xs = [i / 20 * 8 / 255 * tg for i in range(20)]
    for i in range(4):
        losses = []
        with torch.no_grad():
            for j in range(20):
                losses.append(func(x + krylov[-1] * 8 / 255 * tg * (j / 20)))
        v = safe_jac(x + krylov[-1] * 8 / 255 * tg) - b
        for j, u in enumerate(krylov):
            H[:, j, i] = batch_dot(v, u)
            v = v - H[:, j, i].reshape(-1, 1, 1, 1) * u
        H[:, i + 1, i] = torch.linalg.norm(v.flatten(1), dim=-1)
        v = v / H[:, i + 1, i].reshape(-1, 1, 1, 1)
        plotlib.plot(xs, losses, label='Krylov %d' % i)
        krylov.append(v)
    plotlib.ylabel("Cross-entropy", labelpad=16)
    plotlib.xlabel("Perturbation $L_2$-norm", labelpad=16)
    plotlib.legend()
    plotlib.savefig("krylov_c.pdf", bbox_inches="tight")
    plotlib.show()
    if (input() == 'q'):
        break
