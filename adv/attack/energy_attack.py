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
        self.consumed_steps = []

    @property
    def p(self):
        return int((self.basis.shape[0] // 3) ** 0.5 + 0.5)

    def do_clamp(self, x_adv, x):
        x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    def generate_new(self, x_adv, x, step):
        n_pert = self.n_pert
        # return self.do_clamp(x_adv + torch.sign(torch.randn_like(x_adv)) * self.epsilon / 2, x)
        po = 0.5 * step / self.perturb_steps
        # po = 1.0 * step / self.perturb_steps
        k = numpy.random.choice(
            len(self.eigv),
            size=[len(x)],
            p=(self.eigv ** po) / (self.eigv ** po).sum()
        )
        directions = self.basis.T[k]
        sp_dir = numpy.zeros(x_adv.shape)
        nor_dir = numpy.sign(directions) * self.epsilon * 2
        ss = numpy.sign(numpy.random.randn(len(nor_dir), n_pert))
        for m in range(n_pert):
            px, py = numpy.random.randint(x.shape[-1] - self.p, size=[2])
            sp_dir[..., px: px + self.p, py: py + self.p] = (
                ss[:, m].reshape(-1, 1, 1, 1) *
                nor_dir.reshape(-1, 3, self.p, self.p)
            )
        sp_dir = x_adv.new_tensor(sp_dir)
        return self.do_clamp(x_adv + sp_dir, x)

    def change_base(self, step):
        if step == -1:
            self.eigv, self.basis, _ = pickle.load(
                open("./data/attacked-pickle/model2_fw_5-pca.pkl", "rb")
            )

    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach().clone()
        self.change_base(-1)
        threshold = 0.06
        cnt_hit = 1
        cnt_cur = 0
        self.n_pert = int((x.shape[-1] / self.p) ** 2) * 2
        # for _ in range(int((x.shape[-1] / self.p) ** 2 / 2 + 1)):
        for _ in range(10):
            x_adv = self.generate_new(x_adv, x, 0)
        cor, cur = self.per_sample_adamp_loss(model, x_adv, y)
        steps = numpy.full([len(x)], self.perturb_steps + 1)
        for j, suc in enumerate(cor):
            if suc:
                steps[j] = 0
        with torch.no_grad():
            for s in range(self.perturb_steps):
                self.change_base(s)
                active_indices = torch.BoolTensor(cor == False).to(x.device)
                new_samples = self.generate_new(
                    x_adv[active_indices], x[active_indices], s)
                ncr, nex = self.per_sample_adamp_loss(
                    model, new_samples, y[active_indices])
                n_move = 0
                for j, c, n, ns in zip(numpy.arange(len(x))[~cor], cur[active_indices], nex, new_samples):
                    if n > c:
                        x_adv[j] = ns
                        cur[j] = n
                        n_move += 1
                old_ncor = numpy.sum(cor)
                p_move = n_move / (len(x) - old_ncor)
                for j, suc, ns in zip(numpy.arange(len(x))[~cor], ncr, new_samples):
                    if suc:
                        steps[j] = min(steps[j], s + 1)
                        cor[j] = True
                        x_adv[j] = ns
                if p_move < threshold:
                    cnt_cur += 1
                    if cnt_cur >= cnt_hit:
                        threshold *= 2 / 3
                        op = self.n_pert
                        self.n_pert = max(1, int(self.n_pert / 4 + 0.5))
                        if (op != self.n_pert):
                            print(op, "->", self.n_pert)
                        cnt_cur = 0
                else:
                    cnt_cur = 0
                if numpy.sum(cor) != old_ncor:
                    print(s, numpy.sum(cor))
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
            sum(w ** 2 * F.cross_entropy(logits * w, y, reduction='none')
                for w in L)
        )
