import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn-white')
np.random.seed(1919810)


def get_perturb_pkl_name(model_name, attacker_name):
    prefix = './pkls/pics_ptb_'
    return prefix + model_name + '_' + attacker_name + '.pkl'


def get_figure_title(model_name, attacker_name, patch_size):
    patch_size_str = str(patch_size)
    return ' '.join([model_name, attacker_name, patch_size_str]).upper()


def get_save_file_name(model_name, attacker_name, patch_size):
    patch_size_str = str(patch_size)
    return '_'.join([model_name, attacker_name, patch_size_str])


def patching(perturbs, patch_size, downsample=False):
    patch_area = patch_size ** 2
    N, C, H, W = perturbs.shape
    counter = H - patch_size + 1
    if downsample:
        rows = np.random.choice(range(counter), int(counter * 0.09))
        cols = np.random.choice(range(counter), int(counter * 0.09))
    else:
        rows = range(counter)
        cols = range(counter)
    print(f'  - Generating {len(rows)} x {len(cols)} patches.')
    patches = [
        perturbs[item, :, row:row+patch_size, col:col+patch_size].reshape(
            patch_area * C)
        for item in range(N)
        for row in rows
        for col in cols]

    return np.array(patches)


def patch_pca(patches, pca_threshold):
    # centering
    patches_centered = patches - patches.mean()

    # eigenvalue decom
    covar_mat = np.matmul(patches_centered.T, patches_centered)
    value, vector = np.linalg.eig(covar_mat)

    # Get eigenvectors
    sorted_value = np.flip(np.sort(value))
    sorted_idx = np.flip(np.argsort(value))
    cum_sum = np.cumsum(sorted_value)
    cum_sum = cum_sum / cum_sum.max()

    candidates = []
    for i, idx in enumerate(sorted_idx):
        candidates.append(idx)
        if cum_sum[i] > pca_threshold:
            break
    reduced = vector[:, candidates].T  # k, HWC

    return cum_sum, reduced, value, vector


def get_head_patches(reduced, patch_size):
    # Patch, C, H, W
    # each row is a flattened patch
    patch_area = patch_size ** 2
    # # overwrite
    # patch_area = 16 * 16
    head = reduced[:patch_area, :].reshape(
        (patch_area, 3, patch_size, patch_size))
    # head = np.random.rand(3, 5, 5, 25)
    head = np.transpose(head, (0, 2, 3, 1))  # Patch, H, W, C

    return head


def visualize_head_patches(head, patch_size, fig_title, figsize=10):
    head -= head.min()
    head /= head.max()
    # # overwrite
    # patch_size = 16
    fig, axes = plt.subplots(
        patch_size, patch_size, figsize=(figsize, figsize))
    fig.suptitle(fig_title + ' Patches', fontsize=28)
    for i in range(patch_size):
        for j in range(patch_size):
            ax = axes[i, j]
            ax.imshow(head[patch_size * i + j, :, :, :])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([0])
            ax.set_yticks([0])

    return fig, axes


def visualize_cumsum_energy(cum_sum, fig_title, pca_threshold):
    fig, ax = plt.subplots()
    ax.plot(cum_sum, label='Cumulative Energy')
    ax.axhline(pca_threshold, color='r', ls=':', label='PCA Threshold')
    ax.legend()
    ax.set_title(fig_title + ' Cumulative Energy')
    ax.set_ylim([0, 1.05])

    return fig, ax


def load_head_patches(pkl_path):
    head = pickle.load(open(pkl_path, 'rb'))
    return head.reshape(head.shape[0], -1)


def load_all_heads(model_names, attacker_name, patch_size):
    mat = np.zeros((0, patch_size**2 * 3))
    for model in model_names:
        pkl_path = './pkls/' + \
            get_save_file_name(model, attacker_name, patch_size) + \
            '.pkl'
        current = load_head_patches(pkl_path)
        print(f'  - Loaded {pkl_path}')
        mat = np.concatenate((mat, current), axis=0)

    return mat


def generate_block_random_patch(patch_area):
    # overwrite
    # block_rand_patches = np.ones((256, 3*patch_area))
    block_rand_patches = np.ones((patch_area, 3*patch_area))
    red = np.random.randn(block_rand_patches.shape[0], 1)
    green = np.random.randn(block_rand_patches.shape[0], 1)
    blue = np.random.randn(block_rand_patches.shape[0], 1)
    block_rand_patches[:, :patch_area] = \
        block_rand_patches[:, :patch_area] * red
    block_rand_patches[:, patch_area:2 * patch_area] = \
        block_rand_patches[:, patch_area:2*patch_area] * green
    block_rand_patches[:, 2*patch_area:] = \
        block_rand_patches[:, 2*patch_area:] * blue

    block_rand_patches = block_rand_patches / \
        np.expand_dims(
            np.sqrt(np.sum(block_rand_patches ** 2, axis=1)), axis=1)

    return block_rand_patches


def generage_random_patch(patch_area):
    # overwrite
    # rand_patches = np.random.randn(256, 3*patch_area)
    rand_patches = np.random.randn(patch_area, 3*patch_area)
    rand_patches = rand_patches / \
        np.expand_dims(np.sqrt(np.sum(rand_patches ** 2, axis=1)), axis=1)

    return rand_patches


def compute_dot_products(model_names, attacker_name, patch_size):
    patch_area = patch_size ** 2
    all_data = load_all_heads(model_names, attacker_name, patch_size)

    # Generate Block Random Patches
    print('  - Generating Random Patches')
    block_rand_patches = generate_block_random_patch(patch_area)

    # Generate Random Patches
    rand_patches = generage_random_patch(patch_area)

    # Dot Product by Matmul
    print('  - Mafsing')
    all_data = np.concatenate((
        all_data,
        rand_patches,
        block_rand_patches,), axis=0)
    dot_prod = np.abs(np.matmul(all_data, all_data.T))

    return dot_prod


def run_patching_pipeline(
        model_name,
        attacker_name,
        patch_size,
        pca_threshold=0.95):
    pkl_path = get_perturb_pkl_name(model_name, attacker_name)
    fig_title = get_figure_title(model_name, attacker_name, patch_size)
    save_file_name = get_save_file_name(model_name, attacker_name, patch_size)

    # load perturb
    print(f'  - Loading Perturbation from {pkl_path}')
    perturbs = np.array(pickle.load(open(pkl_path, 'rb')))

    # patching
    print('  - Patching')
    patches = patching(perturbs, patch_size, downsample=True)

    # pca
    print('  - Running PCA')
    cum_sum, reduced, value, vector = patch_pca(patches, pca_threshold)

    # save files
    head = get_head_patches(reduced, patch_size)
    pickle.dump(head, open('./pkls/' + save_file_name + '.pkl', 'wb'))
    # pickle.dump(cum_sum, open(save_file_name + '-cumsum.pkl', 'wb'))
    print(f'  - Head samples saved to {"./pkls/" + save_file_name}')
    pickle.dump(
        [value, vector, reduced],
        open('./pkls/' + save_file_name + '-full.pkl', 'wb'))
    print(
        '  - Full results dumped to '
        f'{"./pkls/" + save_file_name + "-full.pkl"}')

    # visualization
    fig_patches, _ = visualize_head_patches(
        head, patch_size, fig_title, figsize=10)
    fig_patches.savefig('./figs/' + save_file_name + '-patches.png', dpi=200)
    fig_cumsum, _ = visualize_cumsum_energy(
        cum_sum, fig_title, pca_threshold)
    fig_cumsum.savefig('./figs/' + save_file_name + '-energy.png', dpi=200)
    print('  - Figures saved')


def visualize_dot_products(dot_product, model_names, patch_size):
    print('  - Drawing Again.')
    patch_area = patch_size ** 2
    ax_mm_label = [
        *model_names,
        'Random',
        'Block Random'
    ]
    ax_mm_ticks = np.arange(0, dot_product.shape[0], patch_area)
    fig_mm, ax_mm = plt.subplots(figsize=(15, 15))
    dotprod_img = ax_mm.imshow(dot_product, cmap='YlGn')
    ax_mm.set_xticks(ax_mm_ticks-0.5)
    ax_mm.set_yticks(ax_mm_ticks-0.5)
    ax_mm.set_xticklabels(ax_mm_ticks)
    ax_mm.set_yticklabels(ax_mm_ticks)
    ax_mm.grid(which='major', axis='both', lw=1, color='k', alpha=0.5, ls='-')
    ax_mm.set_xticks(ax_mm_ticks + patch_area/2, minor=True)
    ax_mm.set_yticks(ax_mm_ticks + patch_area/2, minor=True)
    ax_mm.set_xticklabels(ax_mm_label, minor=True)
    ax_mm.set_yticklabels(ax_mm_label, minor=True)
    plt.setp(
        ax_mm.get_xticklabels(minor=True),
        rotation=45,
        ha="right",
        rotation_mode="anchor")
    fig_mm.colorbar(dotprod_img, ax=ax_mm)
    ax_mm.set_title(f'Dot Products (Patch Size {patch_size})')
    fig_mm.savefig(f'./figs/dot_product_{patch_size}.png', dpi=200)
    print('  - Figure Saved.')

    return fig_mm, ax_mm


def main():
    patch_size = 60
    model_names = ['vgg16bn']
    pca_threshold = 0.95
    attacker_name = 'fw'

    for model in model_names:
        print(f'Running {model}')
        run_patching_pipeline(model, attacker_name, patch_size, pca_threshold)
        print(f'Done Running {model}')

    print('Computing dot products')
    dot_prod = compute_dot_products(model_names, attacker_name, patch_size)
    # # overwrite
    # patch_size = 16
    visualize_dot_products(dot_prod, model_names, patch_size)


if __name__ == '__main__':
    main()
    # print('Yooo.')
