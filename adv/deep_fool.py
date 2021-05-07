import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class P(nn.Module):
    def __init__(self, x_adv):
        super().__init__()
        self.params = nn.Parameter(x_adv)


class DeepFoolAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        for idx, (sample, label) in enumerate(zip(x_adv, y)):
            backup = sample.unsqueeze(0).detach().clone()
            sample = sample.unsqueeze(0).detach().clone().requires_grad_(True)
            k_i = label
            i = 0
            while k_i == label and i < self.perturb_steps:
                predictions = model(sample)  # shape: 1 * num_classes
                num_classes = predictions.shape[-1]
                pertub = float('inf')
                all_gradient = torch.autograd.functional.jacobian(
                    lambda x: model(x), sample)[0]
                original_gradient = all_gradient[label]

                w_argmin = torch.zeros_like(original_gradient)
                for k in range(num_classes):
                    if k == label:
                        continue
                    current_gradient = all_gradient[k]
                    w_k = current_gradient - original_gradient
                    f_k = predictions[0, k] - predictions[0, label]

                    candidate = abs(f_k.item()) / \
                        torch.linalg.norm(w_k.flatten(), 1)
                    if candidate < pertub:
                        pertub = candidate
                        w_argmin = w_k

                pertub = pertub * w_argmin / \
                    torch.linalg.norm(w_argmin.flatten(), 1)

                pertubed_sample = sample.clone().detach() + self.step_size * \
                    torch.sign(pertub.detach())
                pertubed_sample = torch.min(
                    torch.max(pertubed_sample, backup - self.epsilon), backup + self.epsilon)
                pertubed_sample = torch.clamp(pertubed_sample, 0.0, 1.0)

                sample = pertubed_sample.clone().detach().requires_grad_(True)

                predictions = model(sample)
                k_i = torch.argmax(predictions)

                x_adv[idx] = pertubed_sample[0]

                i += 1

        print("#### Attack finished. l-inf Norm: {:.5f}".format(torch.linalg.norm((x - x_adv).flatten(), float('inf')).item()))

        return x_adv
