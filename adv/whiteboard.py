# %%
import cv2 as cv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from perturb_pca import get_perturb_pkl_name, get_save_file_name, load_head_patches


def non_flattening_patching(perturbs, patch_size, downsample=False):
    N, C, H, W = perturbs.shape
    counter = H - patch_size + 1
    if downsample:
        rows = np.random.choice(range(counter), int(counter * 0.15))
        cols = np.random.choice(range(counter), int(counter * 0.15))
    else:
        rows = range(counter)
        cols = range(counter)
    print(f'  - Generating {len(rows)} x {len(cols)} patches.')
    patches = [
        perturbs[item, :, row:row+patch_size, col:col+patch_size]
        for item in range(N)
        for row in rows
        for col in cols]

    return np.array(patches)


def convert_image_to_unit8(img):
    return np.array(8 * (
        np.transpose(img, (1, 2, 0)) + 8/255) * 256, dtype=np.uint8)


def AI2614(img):
    red = cv.equalizeHist(img[:, :, 0])
    green = cv.equalizeHist(img[:, :, 1])
    blue = cv.equalizeHist(img[:, :, 2])
    return np.transpose(np.array([red, green, blue]), (1, 2, 0))


# %%
model = 'model2'
attacker = 'fw'

ptbname = get_perturb_pkl_name(model, attacker)

print(f'  - Loading Perturbation from {ptbname}')
perturbs = np.array(pickle.load(open(ptbname, 'rb')))

# patches = non_flattening_patching(perturbs, 5, downsample=False)

# %%
pkl_path = './pkls/' + \
    get_save_file_name(model, attacker, 5) + \
    '.pkl'
current = load_head_patches(pkl_path)
print(f'  - Loaded {pkl_path}')
current = current.reshape((-1, 3, 5, 5))
current = np.transpose(current, (0, 2, 3, 1))
current = current - current.min()
current = current / current.max()
