import torch
import torch.jit
import torch.nn.functional as F
from tqdm import trange


class ProofOps:
    def __init__(self):
        pass

    def _convolution(self):
        pass

    def batch_norm(self):
        pass

    def relu_(self, *args, **kwargs):
        return self.relu(*args, **kwargs)

    def add(self):
        pass

    def add_(self, *args, **kwargs):
        return self.add(*args, **kwargs)

    def relu(self):
        pass

    def avg_pool2d(self):
        pass

    def view(self):
        pass

    def addmm(self):
        pass


proof_ops = ProofOps()
Tensor = torch.Tensor


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
    func_defs['forward']


def run_model(model, test_loader, device):
    with trange(10000) as progbar:
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)