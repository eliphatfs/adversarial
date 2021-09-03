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


def test_similarity(bas, tar, en_bas, en_tar, nmb):
    q = []
    for i in range(len(tar)):
        for j in range(len(bas)):
            pick = j
            # pick = np.argmax([abs(tar[:, i].dot(np.sign(bas[:, j]))) for j in range(len(bas))])
            q.append(abs(tar[:, i].dot(np.sign(bas[:, pick]))) * en_tar[i] / en_tar.sum() * en_bas[pick] / en_bas.sum())
    return np.sum(q)
    '''A = a.T.astype(np.float64) @ np.diag(ena) @ a.astype(np.float64)
    B = b.T.astype(np.float64) @ np.diag(enb) @ b.astype(np.float64)
    C = (
        a.T.astype(np.float64) @ np.diag(1 / ena) @ a.astype(np.float64)
        @ b.T.astype(np.float64) @ np.diag(enb) @ b.astype(np.float64)
    )
    print(np.linalg.det(A), np.linalg.det(B), np.linalg.det(C))
    return 0.5 * (
        -np.log(np.linalg.det(C)) - len(ena)
        + np.trace(C)
    )'''

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
    # 'CIFAR_RPCONV',
    # 'MNIST_CNN',
    # 'MNIST_DNN',
    # 'MNIST_DNN03',
    # 'MNIST_RPDENSE',
    # 'MNIST_RMDENSE',
    # 'M104',
    # 'C104',
    # 'RMCONV',
    # 'RPCONV',
    'VGG16BN',
    'VITB',
    'VITB_LARGE',
    # 'RANDOM',
    'BLOCK_RANDOM',
]
n_patches = 25
weighted = True
mode = 'least_squares'
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
        else:
            raise ValueError(f'Unsupported mode: {mode}')
        pairwise_cos_similarities[row, col] = test_similarity(basis, target, b_val, t_val, basis_name)

# %%
# plot
matplotlib.style.use('seaborn-white')
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(
    pairwise_cos_similarities,  # / np.nanmax(pairwise_cos_similarities[pairwise_cos_similarities != np.inf]),
    ax=ax,
    annot=True,
    fmt='.4f',
    annot_kws={
        'fontsize': 'medium'
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
