import torch
import numpy
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

    def generate_new(self, x_adv, x, step):
        n_pert = self.n_pert
        noanneal = any('ea:annealoff' in x for x in sys.argv)
        if noanneal:
            if not self.verbosed:
                print("Variant: anneal off")
                self.verbosed = True
            po = 1.0
        else:
            po = 0.5 * step / self.perturb_steps
        k = numpy.random.choice(
            len(self.eigv),
            size=[len(x)],
            p=(self.eigv ** po) / (self.eigv ** po).sum()
        )
        directions = self.basis.T[k]
        sp_dir = numpy.zeros(x_adv.shape)
        nor_dir = numpy.sign(directions).reshape(-1, 3, self.p, self.p) \
            * self.epsilon * 2
        if n_pert > 0:
            nor_dir = numpy.tile(nor_dir, [1, 1, n_pert, n_pert])
        else:
            cx, cy = numpy.random.randint(self.p - 3, size=[2])
            nor_dir = nor_dir[..., cx: cx + 3, cy: cy + 3]
        p = nor_dir.shape[-1]
        ss = numpy.sign(numpy.random.randn(len(nor_dir), 1))
        # for m in range(n_pert):
        px, py = numpy.random.randint(x.shape[-1] - p, size=[2])
        sp_dir[..., px: px + p, py: py + p] = (
            ss[:, 0].reshape(-1, 1, 1, 1) *
            nor_dir
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
        threshold = 0.03
        cnt_hit = 1
        cnt_cur = 0
        self.n_pert = max(1, x.shape[-1] // self.p)
        x_adv = torch.clamp(
            x + self.epsilon * torch.sign(
                torch.randn([3, 1, x.shape[-1]], device=x.device)
            ),
            0.,
            1.)
        cor, cur = self.margin_loss(model, x_adv, y)
        steps = numpy.full([len(x)], self.perturb_steps + 1)
        for j, suc in enumerate(cor):
            if suc:
                steps[j] = 0
        for s in range(self.perturb_steps):
            self.change_base(s)
            active_indices = torch.BoolTensor(cor == False).to(x.device)
            new_samples = self.generate_new(
                x_adv[active_indices], x[active_indices], s)
            ncr, nex = self.margin_loss(
                model, new_samples, y[active_indices])
            n_move = 0
            for j, c, n, ns in zip(numpy.arange(len(x))[~cor], cur[active_indices], nex, new_samples):
                if n > c:
                    x_adv[j] = ns
                    cur[j] = n
                    n_move += 1
            old_ncor = numpy.sum(cor)
            for j, suc, ns in zip(numpy.arange(len(x))[~cor], ncr, new_samples):
                if suc:
                    steps[j] = min(steps[j], s + 1)
                    cor[j] = True
                    x_adv[j] = ns
            p_move = n_move / (len(x) - old_ncor)
            if p_move < threshold:
                cnt_cur += 1
                if cnt_cur >= cnt_hit and self.n_pert > 0:
                    # threshold *= 2 / 3
                    op = self.n_pert
                    if op == 1:
                        self.n_pert = 0
                    else:
                        self.n_pert = max(1, int(self.n_pert / 2 + 0.5))
                    if (op != self.n_pert):
                        print(op, "->", self.n_pert)
                    cnt_cur = 0
                    if self.n_pert == 1:
                        cnt_hit = 20
                        threshold = 0.01
            else:
                cnt_cur = 0
            if numpy.sum(cor) != old_ncor:
                print(s, numpy.sum(cor))
            if numpy.all(ncr):
                break
        self.consumed_steps.extend(steps)
        return torch.clamp(x_adv, 0, 1)

    def margin_loss(self, model, x, y):
        x_rg = x.detach().clone()
        logits = model(x_rg)
        return (
            (logits.max(-1)[-1] != y).cpu().numpy(),
            torch.sort(logits, descending=True)[
                0][..., 1] - torch.sort(logits, descending=True)[0][..., 0]
        )
