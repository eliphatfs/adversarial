import torch
import numpy
import torch.nn.functional as F
import pickle


class EnergyAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.eigv, self.basis, _ = pickle.load(
            open("data/attacked-pickle/model2_fw_9-pca.pkl", "rb")
        )

    def do_clamp(self, x_adv, x):
        x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def generate_new(self, x_adv, x):
        # return self.do_clamp(x_adv + torch.sign(torch.randn_like(x_adv)) * self.epsilon / 2, x)
        k = numpy.random.choice(
            len(self.eigv),
            size=[len(x)],
            p=self.eigv / self.eigv.sum()
        )
        directions = self.basis.T[k]
        sp_dir = numpy.zeros(x_adv.shape)
        nor_dir = numpy.sign(directions) * self.epsilon * 2
        for j, patch in enumerate(nor_dir):
            for _ in range(3):
                px, py = numpy.random.randint(32 - 9, size=[2])
                s = numpy.sign(numpy.random.randn())
                sp_dir[j, :, px: px + 9, py: py + 9] += s * patch.reshape(3, 9, 9)
        sp_dir = x_adv.new_tensor(sp_dir)
        return self.do_clamp(x_adv + sp_dir, x)

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        for _ in range(int((32 / 9) ** 2)):
            x_adv = self.generate_new(x_adv, x)
        cur = self.per_sample_adamp_loss(model, x_adv, y)
        with torch.no_grad():
            for _ in range(self.perturb_steps):
                new_samples = x_adv
                new_samples = self.generate_new(x_adv, x)
                nex = self.per_sample_adamp_loss(model, new_samples, y)
                for j, (c, n) in enumerate(zip(cur, nex)):
                    if n > c:
                        x_adv[j] = new_samples[j]
                        cur[j] = n
        return torch.clamp(x_adv, 0, 1)

    def per_sample_adamp_loss(self, model, x, y):
        x_rg = x.detach().clone()
        logits = model(x_rg)
        L = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return sum(w ** 2 * F.cross_entropy(logits * w, y, reduction='none') for w in L)
