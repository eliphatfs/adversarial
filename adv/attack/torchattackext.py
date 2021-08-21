from torchattacks.attacks.square import Square
import torch


class TAEXT():
    def __init__(self, step_size, epsilon, perturb_steps, sub,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.square = None
        self.real_batch_size = sub

    def __call__(self, model, x, y):

        def call_model(x_rg):
            subbatches = torch.split(x_rg, len(x_rg) // self.real_batch_size + 1)
            return torch.cat([model(sb) for sb in subbatches])

        if self.square is None:
            self.square = Square(
                call_model, n_queries=self.perturb_steps,
                eps=self.epsilon, verbose=True
            )
        with torch.no_grad():
            return self.square(x, y)
