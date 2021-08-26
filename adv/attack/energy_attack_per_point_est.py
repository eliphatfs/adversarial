import torch
import numpy
import torch.nn.functional as F
import pickle
import sys


class EnergyAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start
        self.consumed_steps = []
        self.verbosed = False

    @property
    def p(self):
        return int((self.basis.shape[0] // 3) ** 0.5 + 0.5)

    def do_clamp(self, x_adv, x):
        x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def generate_new(self, x_adv, x, step, n_perts):
        # return self.do_clamp(x_adv + torch.sign(torch.randn_like(x_adv)) * self.epsilon / 2, x)
        noanneal = any('ea:annealoff' in x for x in sys.argv)
        if noanneal:
            if not self.verbosed:
                print("Variant: anneal off")
                self.verbosed = True
            po = 1.0
        else:
            po = 0.5 * step / self.perturb_steps
        # po = 1.0 * step / self.perturb_steps
        k = numpy.random.choice(
            len(self.eigv),
            size=[len(x)],
            p=(self.eigv ** po) / (self.eigv ** po).sum()
        )
        directions = self.basis.T[k].astype(numpy.float32)
        sp_dir = numpy.zeros(x_adv.shape, dtype=numpy.float32)
        nor_dir = numpy.sign(directions).reshape(-1, 3, self.p, self.p) * (self.epsilon * 2)
        for n_pert in set(n_perts):
            sel_dir = nor_dir[n_perts == n_pert]
            if n_pert > 0:
                tile_dir = numpy.tile(sel_dir, [1, 1, n_pert, n_pert])
            else:
                cx, cy = numpy.random.randint(self.p - 3, size=[2])
                tile_dir = sel_dir[..., cx: cx + 3, cy: cy + 3]
            p = tile_dir.shape[-1]
            ss = numpy.sign(numpy.random.randn(len(tile_dir)))
            px, py = numpy.random.randint(x.shape[-1] - p, size=[2])
            sp_dir[n_perts == n_pert, ..., px: px + p, py: py + p] = (
                ss.reshape(-1, 1, 1, 1) *
                tile_dir
            )
        return self.do_clamp(x_adv + x_adv.new_tensor(sp_dir), x)

    def change_base(self, step):
        overwrite_pca = 'model2_fw_5-pca.pkl'
        for x in sys.argv:
            sp = x.split('ea:basepkl:')
            if len(sp) > 1:
                overwrite_pca = sp[-1]
        if step == -1:
            self.eigv, self.basis, _ = pickle.load(
                open("./data/attacked-pickle/" + overwrite_pca, "rb")
            )

    def __call__(self, model, x, y):
        with torch.no_grad():
            return self.call_internal(model, x, y)

    def call_internal(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        self.change_base(-1)
        x_adv = torch.clamp(x + self.epsilon * torch.sign(
                            torch.randn([3, 1, x.shape[-1]], device=x.device)), 0., 1.)
        cor, cur = self.per_sample_adamp_loss(model, x_adv, y)
        steps = numpy.full([len(x)], self.perturb_steps + 1)
        stuck = numpy.full([len(x)], 0)
        n_pert = numpy.full([len(x)], max(1, x.shape[-1] // self.p))
        for j, suc in enumerate(cor):
            if suc:
                steps[j] = 0
        for s in range(self.perturb_steps):
            self.change_base(s)
            active_indices = torch.BoolTensor(cor == False).to(x.device)
            new_samples = self.generate_new(
                x_adv[active_indices], x[active_indices], s, n_pert[cor == False])
            ncr, nex = self.per_sample_adamp_loss(
                model, new_samples, y[active_indices])
            stuck = stuck + 1
            for j, c, n, ns in zip(numpy.arange(len(x))[~cor], cur[active_indices], nex, new_samples):
                if n > c:
                    x_adv[j] = ns
                    cur[j] = n
                    stuck[j] = 0
            n_pert[(n_pert == 1) & (stuck >= 7)] = 0
            n_pert[stuck >= 15] = n_pert[stuck >= 15] / 2 + 0.5
            stuck[stuck >= 15] = 0
            old_ncor = numpy.sum(cor)
            for j, suc, ns in zip(numpy.arange(len(x))[~cor], ncr, new_samples):
                if suc:
                    steps[j] = min(steps[j], s + 1)
                    cor[j] = True
                    x_adv[j] = ns
            if numpy.sum(cor) != old_ncor:
                print(s, numpy.sum(cor), '%.1f' % numpy.median(n_pert))
            if numpy.all(ncr):
                break
        self.consumed_steps.extend(steps)
        return torch.clamp(x_adv, 0, 1)

    def per_sample_adamp_loss(self, model, x, y):
        x_rg = x.detach().clone()
        logits = model(x_rg)
        L = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        return (
            (logits.max(-1)[-1] != y).cpu().numpy(),
            torch.sort(logits, descending=True)[0][..., 1] - torch.sort(logits, descending=True)[0][..., 0]
            # sum(w ** 2 * F.cross_entropy(logits * w, y, reduction='none')
            #     for w in L)
        )
