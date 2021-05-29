import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import OrderedDict

EPSILON = 1e-20


class AdvWeightPerturb(object):
    def __init__(
            self,
            model: nn.Module,
            proxy: nn.Module,
            proxy_optim: optim.Optimizer,
            gamma,):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        print(
            "Adversarial Weight Perturbation armed and ready!", file=sys.stderr)

    def _normalize(self, diff: torch.Tensor, weight: torch.Tensor):
        return diff * weight.norm() / (diff.norm() + EPSILON)

    def _add_pertubation(self, coeff=1.0):
        keys = self.diff_dict.keys()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in keys:
                    param.add_(coeff * self.diff_dict[name])

    def _calc_weight_diff(self):
        diff_dict = OrderedDict()
        model_state_dict = self.model.state_dict()
        proxy_state_dict = self.proxy.state_dict()

        for (old_key, old_weight), (new_key, new_weight) \
                in zip(model_state_dict.items(), proxy_state_dict.items()):
            if len(old_weight.size()) <= 1:
                continue  # ignore 1d weights
            if 'weight' in old_key:
                diff_weight = new_weight - old_weight
                diff_dict[old_key] = self._normalize(
                    diff_weight, old_weight)

        return diff_dict

    def calc_awp(self, x_adv: torch.Tensor, y: torch.Tensor):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        pred = self.proxy(x_adv)
        negative_loss = - F.cross_entropy(pred, y)

        self.proxy_optim.zero_grad()
        negative_loss.backward()
        self.proxy_optim.step()

        self.diff_dict = self._calc_weight_diff()

        return self.diff_dict

    def perturb_model(self):
        self._add_pertubation(self.gamma * 1.0)

    def restore_perturb(self):
        self._add_pertubation(self.gamma * -1.0)
