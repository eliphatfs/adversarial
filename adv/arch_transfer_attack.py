import torch
import numpy
import torch.nn.functional as F
from typing import *


class MonkeyPatcher:
    def __init__(self):
        self.relu_ = self.relu

    def relu(self, x):
        return F.leaky_relu(x, 0.2)

    def __getattr__(self, attr):
        try:
            return getattr(torch, attr)
        except AttributeError:
            return getattr(torch.Tensor, attr)


proof_ops = MonkeyPatcher()
Tensor = torch.Tensor
torch.avg_pool2d = F.avg_pool2d
torch.view = lambda x, *args: x.view(*args)
annotate = lambda _, x: x


def process_model(model, x):
    global CONSTANTS
    trace_module = torch.jit.trace(model, x)
    del trace_module.training
    trace_module.eval()
    frozen_module = torch.jit.freeze(trace_module)
    code, consts = frozen_module.code_with_constants
    func_defs = dict()
    exec(code.replace("torch.", "proof_ops."), globals(), func_defs)
    CONSTANTS = consts
    return func_defs['forward']


class ArchTransferAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.processed_model_id = None
        self.records = []

    def __call__(self, model, x, y):
        model.eval()
        if self.processed_model_id != id(model):
            self.processed_model = process_model(model, x)
        x_adv = x.detach()
        # self.records.append((self.processed_model(None, x).argmax(-1) == y).float().mean().item())
        # print(numpy.mean(self.records))
        for i in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(self.processed_model(None, x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv