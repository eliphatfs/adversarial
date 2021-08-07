import pickle
import torch
import torch.nn.functional as F
from datetime import datetime


class StochasticFWAdampAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps * 10
        self.random_start = random_start
        self.step_size = epsilon / 20

    def __call__(self, model, x, y):
        return self.fw_spsa(model, x, y)

    def adamp(self, pred, y):
        L = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        adamp_loss = sum(w ** 2 * F.cross_entropy(pred * w, y) for w in L)
        return adamp_loss

    def fw_spsa(self, model, x, y):
        succ_rate = []
        model.eval()
        x_adv = x.detach()
        succeeded_attacks = x.detach()
        with torch.no_grad():
            for k in range(self.perturb_steps):
                perturb = torch.sign(torch.randn_like(x_adv))

                x_plus = x_adv.detach() + perturb.detach()*self.step_size
                x_plus = torch.min(
                    torch.max(x_plus, x - self.epsilon), x + self.epsilon)
                x_plus = torch.clamp(x_plus, 0.0, 1.0)
                y_plus = self.adamp(model(x_plus), y)

                x_minus = x_adv.detach() - perturb.detach()*self.step_size
                x_minus = torch.min(
                    torch.max(x_minus, x - self.epsilon), x + self.epsilon)
                x_minus = torch.clamp(x_minus, 0.0, 1.0)
                y_minus = self.adamp(model(x_minus), y)

                gp = (y_plus - y_minus) / (2*self.step_size)

                s = x + torch.sign(gp * perturb)
                a = 2 / (k + 2)
                x_adv = x_adv + a * (s - x_adv)
                print(f'x: {x.max()}')
                print(f'xadv Before clamping: {x_adv.max()}')
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                print(f'xadv After clamping: {x_adv.max()}')

                y_pred = model(x_adv).argmax(-1)
                batch_successes = torch.zeros_like(y)
                batch_size = x_adv.shape[0]
                for idx, (lab_pred, lab_true) in enumerate(zip(y_pred, y)):
                    if lab_pred != lab_true:
                        succeeded_attacks[idx] = x_adv[idx]
                        # print(
                        #     f'Iter {k}: Updated {idx}. '
                        #     f'Pred_old: {lab_pred} | True: {lab_true}')
                        batch_successes[idx] = 1
                # print(
                #     f'Iter {k}: {batch_successes.sum()} '
                #     f'out of {batch_successes.shape[0]} successes.')
                succ_rate.append(batch_successes.sum() / batch_size)

        # time = datetime.now()
        # fname_by_time = time.strftime('%H_%M_%S')
        # pickle.dump(succ_rate, open(
        #     f'./{fname_by_time}.pkl', 'wb'))
        return succeeded_attacks
