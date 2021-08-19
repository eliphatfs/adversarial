from torchattacks.attacks.square import Square


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
            self.square = Square(model, n_queries=self.perturb_steps, eps=self.epsilon)
        return self.square(x, y)
