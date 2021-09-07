# %%
import seaborn as sns
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from perturb_pca import compute_dot_products
from mpl_toolkits.axes_grid1 import make_axes_locatable


# matplotlib.style.use('seaborn-white')
plt.rcParams['font.family'] = 'Calibri'


def format_axes(
    ax: plt.Axes,
    ticklabels,
    patch_size,
    total_size,
    block_size,
):
    robust_models = [1, 2]
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
    ax.set_xticklabels(ticklabels, minor=True, fontsize=12, weight='light')
    ax.set_yticklabels(ticklabels, minor=True, fontsize=12, weight='light')
    plt.setp(
        ax.get_yticklabels(minor=True),
        rotation=90,
        ha="center",
        va="bottom",
        rotation_mode="anchor")
    # ax.set_title(f'Patch Size {patch_size}', fontsize=1, pad=10)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Models', fontsize=14)
    ax.xaxis.labelpad = 7
    ax.yaxis.labelpad = 7
    # for rm in robust_models:
    #     plt.setp(
    #         ax.get_yticklabels(minor=True)[rm],
    #         style='italic',
    #         weight='bold')
    #     plt.setp(
    #         ax.get_xticklabels(minor=True)[rm],
    #         style='italic',
    #         weight='bold')


# %%
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
    'WRN',
    'DNN',
    'VGG',
    'ViT',
    'RND',
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(
        imobj, cax=cax, ticks=[0, 1]
        # aspect=40, fraction=0.07,
    )

fig.savefig('./figs/dot_product_trinity.pdf', bbox_inches='tight')

# %%
models = [
    'Model2',
    'MNIST_RMDENSE',
    'MNIST_RPDENSE',
    'RMCONV',
    'RPCONV',
]
ticklabels = [
    'R18',
    'RMDNN',
    'RPDNN',
    'RMCNN',
    'RPCNN',
]

fig, ax = plt.subplots()

dot_product = compute_dot_products(models, 'fw', 5)
imobj = ax.imshow(dot_product, cmap='YlGn')
format_axes(ax, ticklabels, 5, 125, 25)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(
    imobj, cax=cax, ticks=[0, 1]
    # aspect=40, fraction=0.07,
)

fig.savefig('./figs/dot_product_MNIST_NEW.pdf', bbox_inches='tight')


# %%
sns.set_theme(context='paper', font='Calibri', font_scale=1.25)
x = [1, 2, 4, 7, 10, 13]
mnist_lenet_eps_0_3 = [45.5921, 50.5076, 54.9649, 69.1903, 69.3305, 79.3606]
imagenet_inception_eps_0_05 = [91.08, 88.42, 101.39, 107.70, 128.35, 140.05]

fig, ax = plt.subplots()

ax = sns.lineplot(x=x, y=mnist_lenet_eps_0_3, label='MNIST LeNet', ax=ax)
ax = sns.lineplot(x=x, y=imagenet_inception_eps_0_05, ax=ax, label='ImageNet InceptionV3')
sns.despine()
ax.set_xlabel('Threshold $\\tau$')
ax.set_ylabel('Avg. Queries')
fig.savefig('./figs/batch_threshold.pdf', bbox_inches='tight')
