import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-white')


def get_cumsum_file_name(model_name, attacker_name, patch_size):
    return './'\
        + model_name + '_'\
        + attacker_name + '_'\
        + str(patch_size) + '-cumsum'


def load_cumsum_pkl(path):
    return pickle.load(open(path, 'rb'))


attacker = 'fw'
model_names = ['model1', 'model2', 'model3', 'model4', 'WRN28']
patch_sizes = [3, 9, 15, 32]

for model in model_names:
    fig, ax = plt.subplots()
    for patch_size in patch_sizes:
        fname = get_cumsum_file_name(model, attacker, patch_size)
        cum_sum = load_cumsum_pkl(
            fname + '.pkl')
        x = np.arange(0, len(cum_sum), 1, dtype=float) + 1
        x /= len(cum_sum)
        ax.plot(x, cum_sum, label=f'Patch size: {patch_size}')
        ax.set_ylabel('Cumulative Energy')
        ax.set_xlabel('$k/N$')
        ax.set_title(model)
        ax.legend()
    fig.savefig('./figs/' + f'{model}_fw-cumsum.png', dpi=200)
