import torch
import torch.optim


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
        x_cpu = x.cpu().detach().clone()
        active_indices = list(range(len(x_adv)))
        for i in range(self.perturb_steps):
            if len(active_indices) == 0:
                break
            samples = x_adv[active_indices]
            predictions = model(samples).cpu()
            all_gradient = torch.autograd.functional.jacobian(
                lambda cx: model(cx).sum(0), samples
            ).transpose(0, 1).cpu()
            for ei, idx in enumerate(active_indices):
                label = y[idx].item()
                sample = samples[ei].cpu()
                num_classes = predictions.shape[-1]
                pertub = float('inf')
                original_gradient = all_gradient[ei, label]

                w_argmin = torch.zeros_like(original_gradient)
                for k in range(num_classes):
                    if k == label:
                        continue
                    current_gradient = all_gradient[ei, k]
                    w_k = current_gradient - original_gradient
                    f_k = predictions[ei, k] - predictions[ei, label]

                    candidate = abs(f_k.item()) / \
                        torch.linalg.norm(w_k.flatten(), 1)
                    if candidate < pertub:
                        pertub = candidate
                        w_argmin = w_k

                pertub = pertub * w_argmin / \
                    torch.linalg.norm(w_argmin.flatten(), 1)

                pertubed_sample = sample + self.step_size * torch.sign(pertub)
                pertubed_sample = torch.min(torch.max(pertubed_sample, x_cpu[idx] - self.epsilon), x_cpu[idx] + self.epsilon)
                pertubed_sample = torch.clamp(pertubed_sample, 0.0, 1.0)

                x_adv[idx] = pertubed_sample.to(x_adv.device)
            predictions = model(x_adv[active_indices])
            pred_labels = predictions.argmax(-1)
            active_indices = [
                idx for idx, ypr, ytr in zip(active_indices, pred_labels, y[active_indices])
                if ypr == ytr
            ]

        return x_adv
