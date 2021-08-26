# %%
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from perturb_pca import compute_dot_products


matplotlib.style.use('seaborn-white')


def format_axes(
    ax: plt.Axes,
    ticklabels,
    patch_size,
    total_size,
    block_size,
):
    ax_mm_ticks = np.arange(0, total_size, block_size)
    ax.set_xticks(ax_mm_ticks-0.5)
    ax.set_yticks(ax_mm_ticks-0.5)
    ax.set_xticklabels(ax_mm_ticks)
    ax.set_yticklabels(ax_mm_ticks)
    ax.grid(which='major', axis='both', lw=1, color='k', alpha=0.5, ls='-')
    ax.set_xticks(ax_mm_ticks + block_size/2, minor=True)
    ax.set_yticks(ax_mm_ticks + block_size/2, minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels(ticklabels, minor=True)
    ax.set_yticklabels(ticklabels, minor=True)
    plt.setp(
        ax.get_yticklabels(minor=True),
        rotation=90,
        ha="center",
        va="bottom",
        rotation_mode="anchor")
    ax.set_title(f'Patch Size {patch_size}', fontsize=14)
    ax.set_xlabel('Models', fontsize=13)
    ax.set_ylabel('Models', fontsize=13)


# %%
# matplotlib.style.use('seaborn-white')

patch_size_configs = [5, 15, 28]
max_pool_kernel = [1, 3, 14]
attacker = 'fw'
models = [
    'Model1',
    'Model2',
    'Model4',
    'MNIST_DNN',
    'VGG16BN',
    'VITB'
]
ticklabels = [
    'R34',
    'R18',
    'WR34',
    'DNN',
    'VGG',
    'ViTB',
    'RAND',
    'SQR'
]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for idx, p_size in enumerate(patch_size_configs):
    ax = axs[idx]
    dot_product = compute_dot_products(models, attacker, p_size)
    downsample = max_pool_kernel[idx]
    if downsample > 1:
        pooled = F.max_pool2d(
            torch.Tensor(dot_product).unsqueeze(0).unsqueeze(0),
            downsample)
        dot_product = pooled.squeeze(0).squeeze(0).numpy()
    imobj = ax.imshow(dot_product, cmap='YlGn')
    format_axes(
        ax,
        ticklabels,
        p_size,
        dot_product.shape[0],
        p_size ** 2 // downsample
    )
fig.colorbar(
    imobj, ax=axs, location='right',
    # aspect=40, fraction=0.07,
)
