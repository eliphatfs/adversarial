# %% Init
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-white')


patch_size = 5
patch_area = patch_size ** 2
pca_threshold = 0.95
pkl_path = './pics_ptb_mdl3_fw.pkl'
title = 'Model3 with FW-AdAmp'

perturbs = np.array(pickle.load(open(pkl_path, 'rb')))

# %% Patching
N, C, H, W = perturbs.shape

counter = H - patch_size + 1

patches = [
    perturbs[item, :, row:row+patch_size, col:col+patch_size].reshape(
        patch_area * C)
    for item in range(N)
    for row in range(counter)
    for col in range(counter)]
patches = np.array(patches)

# %% PCA
# center
patches_centered = patches - patches.mean()

# eigenvalue decom
covar_mat = np.matmul(patches_centered.T, patches_centered)
value, vector = np.linalg.eig(covar_mat)

# Get eigenvectors
sorted_value = np.flip(np.sort(value))
sorted_idx = np.flip(np.argsort(value))
cumsum = np.cumsum(sorted_value)
cumsum = cumsum / cumsum.max()

candidates = []
for i, idx in enumerate(sorted_idx):
    candidates.append(idx)
    if cumsum[i] > pca_threshold:
        break

reduced = vector[:, candidates]  # HWC, k

# %% Visualize Patches
head = reduced[:, :patch_area].reshape(
    (3, patch_size, patch_size, patch_area), order='F')  # C, H, W, Patch
# head = np.random.rand(3, 5, 5, 25)
head = np.transpose(head, (3, 1, 2, 0))  # Patch, H, W, C
head -= head.min()
head /= head.max()
fig, axs = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle(title, fontsize=28)
for i in range(5):
    for j in range(5):
        ax = axs[i, j]
        ax.imshow(head[5 * i + j, :, :, :])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0])
        ax.set_yticks([0])
fig.savefig('./figs/' + title + '.png', dpi=200)

# %% Visualize Energy
fig_cumsum, ax_cumsum = plt.subplots()
ax_cumsum.plot(cumsum, label='Cumulative Energy')
ax_cumsum.legend()
ax_cumsum.axhline(pca_threshold, color='r', ls=':')
ax_cumsum.set_title(title + ' Cumulative Energy')
ax_cumsum.set_ylim([0, 1.05])
fig_cumsum.savefig('./figs/' + title + '-energy.png', dpi=200)
