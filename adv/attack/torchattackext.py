from torchattacks.attacks.square import Square
import torch


class TAEXT():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.square = None

    def __call__(self, model, x, y):
        if self.square is None:
            self.square = Square(
                model, n_queries=self.perturb_steps,
                eps=self.epsilon, verbose=True, p_init=0.05 if x.shape[-1] > 200 else 0.8
            )
        with torch.no_grad():
            return self.square(x, y)
