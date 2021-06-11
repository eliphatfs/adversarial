import numpy
import torch
import torch.nn.functional as F
import torch.jit
import diffai.ai
from tqdm import trange
import argparse
from typing import *
from model import get_model_for_attack
from utils import get_test_cifar


class ProofOps:
    def __init__(self):
        pass

    @staticmethod
    def _convolution_wrap(
        tensor, weight, bias,
        stride, padding, dialation,
        transposed, output_pad, groups,
        benchmark, deterministic, cudnn_en
    ):
        return torch._convolution(
            tensor, weight, bias,
            stride, padding, dialation,
            transposed, output_pad, groups,
            benchmark, deterministic, cudnn_en
        )

    def _convolution(
        self, dom, weight, bias,
        stride, padding, dialation,
        transposed, output_pad, groups,
        benchmark, deterministic, cudnn_en, _
    ):
        # assert not transposed, "Deconvolution is not supported yet."
        # assert all(x == 0 for x in output_pad), "Output pad is not supported yet."
        # assert len(stride) == len(padding) == len(dialation) == 2, 'We only implemented conv2d.'
        # assert groups == 1, 'Group convolution is not supported yet.'
        return dom.conv(
            ProofOps._convolution_wrap,
            weight, bias,
            stride=stride, padding=padding, dialation=dialation,
            transposed=transposed, output_pad=output_pad, groups=groups,
            benchmark=benchmark, deterministic=deterministic, cudnn_en=cudnn_en
        )

    def batch_norm(
        self, dom, weight, bias,
        running_mean, running_var,
        training, momentum, eps,
        cudnn_en
    ):
        assert len(
            dom.head.shape) == 4, 'batch_norm is only implemented for 2d data currently.'
        zero_mean = dom - running_mean.unsqueeze(-1).unsqueeze(-1)
        normalized = zero_mean / \
            (running_var.unsqueeze(-1).unsqueeze(-1) + eps).sqrt()
        return normalized * weight.unsqueeze(-1).unsqueeze(-1) + bias.unsqueeze(-1).unsqueeze(-1)

    def avg_pool2d(
        self, dom, kernel_size,
        stride, padding, ceil_mode,
        count_include_pad, divisor_override
    ):
        return dom.avg_pool2d(
            kernel_size, stride, padding,
            ceil_mode, count_include_pad, divisor_override
        )

    def addmm(self, bias, din, weight, beta=1, alpha=1):
        return din.matmul(weight) * alpha + bias * beta

    def add(self, a, b, alpha=1):
        return a + b * alpha

    def relu(self, dom):
        return dom.relu()

    def view(self, dom, shape):
        return dom.view(shape)

    def relu_(self, *args, **kwargs):
        return self.relu(*args, **kwargs)

    def add_(self, *args, **kwargs):
        return self.add(*args, **kwargs)


proof_ops = ProofOps()
Tensor = torch.Tensor
torch.avg_pool2d = F.avg_pool2d
torch.view = lambda x, *args: x.view(*args)


def annotate(_, a):
    return a


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


def run_model(model, test_loader, device, eps):
    model.eval()
    abstract_model = None
    acc = []
    with trange(10000) as progbar, torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if abstract_model is None:
                abstract_model = process_model(model, x)
            abstract_input = diffai.ai.HybridZonotope(
                x, torch.ones_like(x) * 1e-7, None
            ).checkSizes()
            abstract_logits = abstract_model(None, abstract_input)
            print(abstract_logits.head.argmax(-1) == y)
            acc.append(abstract_logits.isSafe(y).float().mean().item())
            progbar.set_description("Provable Acc: %.5f" % numpy.mean(acc))
            progbar.update(x.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--model_name', type=str, default='model6')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epsilon', type=float, default=8/255)
    ns = parser.parse_args()
    model = get_model_for_attack(ns.model_name).to(ns.device)
    test_loader = get_test_cifar(ns.batch_size)
    run_model(model, test_loader, ns.device, ns.epsilon)
