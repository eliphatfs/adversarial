# %%
# Init
import pickle
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from perturb_pca import get_save_file_name
from sklearn.linear_model import LinearRegression


def load_eigens(model_name, attacker_name='fw', patch_size=5):
    pkl_path = get_save_file_name(model_name, attacker_name, patch_size)
    val, vec, _ = pickle.load(open('./pkls/' + pkl_path + '-full.pkl', 'rb'))
    return val, vec


def abs_cosine_similarity(a, b):
    cos_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.abs(cos_similarity)


def maximum_cosine_similarity(basis, targets, n_patches, weight=None):
    results = np.zeros(n_patches)
    for tgt_idx in range(n_patches):
        tgt = targets[:, -(tgt_idx + 1)]

        X = basis[:, -n_patches:]
        maximum = 0
        for base_idx in range(n_patches):
            x = X[:, base_idx]
            current = abs_cosine_similarity(x, tgt)
            if current > maximum:
                maximum = current

        results[tgt_idx] = maximum

    if weight is None:
        return results.mean()
    else:
        return np.dot(weight, results)

    return results.mean()


def least_square_cosine_similarity(basis, targets, n_patches, weight=None):
    results = np.zeros(n_patches)
    for idx in range(n_patches):
        tgt = targets[:, -(idx + 1)]  # an eigenvector of shape (n_samples,)
        least_squares = LinearRegression(fit_intercept=False)

        X = basis[:, -n_patches:]  # n_patches eigenvectors as bases
        least_squares.fit(X, tgt)  # fit tgt vector
        dist = np.matmul(X, least_squares.coef_)
        cos_similarity = np.abs(abs_cosine_similarity(dist, tgt))
        results[idx] = cos_similarity

    if weight is None:
        return results.mean()
    else:
        return np.dot(weight, results)


# %%
# params
patch_size = 5
attacker = 'fw'

model_names = [
    'MODEL1',
    'MODEL2',
    'MODEL3',
    'MODEL4',
    'WRN28',
    'CIFAR_RPCONV',
    'MNIST_CNN',
    'MNIST_DNN',
    'MNIST_RPDENSE',
    'MNIST_RMDENSE',
    'VGG16BN',
    'VITB',
    'VITB_LARGE',
    'RANDOM',
    'BLOCK_RANDOM'
]
n_patches = 25
weighted = True
mode = 'maximum'

mode_title = 'Least Squares' if mode == 'least_squares' else 'Maximum'
weighted_fname = 'weighted' if weighted else ''
weighted_title = ' Weighted' if weighted else ''

# %%
# calc
pairwise_cos_similarities = np.zeros((len(model_names), len(model_names)))
for row, basis_name in enumerate(model_names):
    for col, target_name in enumerate(model_names):
        b_val, basis = load_eigens(basis_name)
        t_val, target = load_eigens(target_name)
        if weighted:
            weights = t_val[-n_patches:]
            weights = weights / weights.sum()
        else:
            weights = None
        if mode == 'least_squares':
            cos_sim = least_square_cosine_similarity(
                basis, target, n_patches, weights)
        elif mode == 'maximum':
            cos_sim = maximum_cosine_similarity(
                basis, target, n_patches, weights)
        pairwise_cos_similarities[row, col] = cos_sim

# %%
# plot
matplotlib.style.use('seaborn-white')
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    pairwise_cos_similarities,
    ax=ax,
    annot=True,
    fmt='.4f',
    annot_kws={
        'fontsize': 'large'
    },
    cmap='YlGn',
    linewidths=0.5,
    square=True,
    xticklabels=model_names,
    yticklabels=model_names,
)

plt.setp(
    ax.get_xmajorticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor")

plt.setp(
    ax.get_ymajorticklabels(),
    rotation=0,
    ha="right",
    rotation_mode="anchor")

ax.set_xlabel('Targets', fontsize=16)
ax.set_ylabel('Bases', fontsize=16)
ax.set_title(
    f'Pairwise{weighted_title} {mode_title} Cosine Similarity', fontsize=24)

fig.savefig(f'./figs/{mode}_cos_sim_{n_patches}{weighted_fname}.pdf')
