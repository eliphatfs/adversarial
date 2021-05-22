import torch
from utils import get_test_cifar
import matplotlib.pyplot as plotlib
from model import get_model_for_attack
import vgg
plotlib.style.use('seaborn')


test_loader = get_test_cifar(1)
model = get_model_for_attack('model5')
model = vgg.vgg13_bn()
ensemble = torch.load('vgg13bn_regm.dat', map_location=torch.device('cpu'))
model.load_state_dict(ensemble)
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
    krylov = [b / torch.norm(b.flatten(1), -1).reshape(-1, 1, 1, 1)]

    batch_dot = lambda u, v: torch.bmm(u.flatten(1).unsqueeze(-2), v.flatten(1).unsqueeze(-1)).flatten(0)
    H = torch.zeros(
        [x.shape[0], 5, 4],
        dtype=x.dtype,
        device=x.device
    )
    xs = [i / 20 for i in range(20)]
    for i in range(4):
        losses = []
        v = safe_jac(x + krylov[-1] * 8 / 255 * tg) - b
        for j, u in enumerate(krylov):
            H[:, j, i] = batch_dot(v, u)
            v = v - H[:, j, i].reshape(-1, 1, 1, 1) * u
        H[:, i + 1, i] = torch.linalg.norm(v.flatten(1), dim=-1)
        v = v / H[:, i + 1, i].reshape(-1, 1, 1, 1)
        with torch.no_grad():
            for j in range(20):
                losses.append(func(x + v * 8 / 255 * tg * (j / 20)))
        plotlib.plot(xs, losses, label='Krylov %d' % i)
        krylov.append(v)
    plotlib.legend()
    plotlib.show()
