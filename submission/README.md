# Energy Attack

> A Parameter-free Transfer-based Black-box L-inf Attack.

## Getting Started

### Environment

All the code, unless otherwise stated, should be able to run with

- pytorch 1.8.1 with cuda 10.2
- numpy 1.19.2
- scipy 1.6.2

### Dataset

- The test set we used for ImageNet is available [here].
- Please unzip the dataset to `./data/`.
- The file structure should be like `./data/images2/ILSVRC2012_val_00000094.JPEG`.

### Patches

- The patches we have used in our paper is available with this code submission.
- Patches should be put under `./data/attacked-pickle/`
  - e.g. `./data/attacked-pickle/model2_fw_5-pca.pkl`
- You can also use other patches, [see here](#using-other-perturbation-patches).

## Running Energy Attack

### Quick Start

You can run Energy Attack on a batch-normalized VGG16 on the ImageNet dataset by running the following in your terminal.

```shell
python attack_main.py --model_name model_vgg16bn --attacker energy --custom_flags ea:annealoff
```

### Arguments for `attack_main.py`

- `--batch_size`: Default 128.
- `--subbatch_size`: A batch is further divided into multiple sub-batches. This argument specifies the size of sub-batches. Default 16.
- `--epsilon`: The maximum L-inf distance for the perturbation. Default 0.05.
- `perturb_steps`: For white-box attacks, this is the maximum step of perturbations. For black-box attacks, this is the maximum number of queries. Default 10000 (for black-box attacks).
- `--model_name`: The model to attack. We currently support:
  - `model_vgg16bn`: batch-normalized VGG16
  - `model_resnet18`: ResNet18
  - `model_inceptionv3`: InceptionV3
  - `model_vitb`: ViTB16
- `--attacker`: The attack to be used. We currently support:
  - `energy`: Energy Attack **(ours)**. (Black-box)
  - `fw`: Frank-Wolfe Attack. (White-box)
- `--custom_flags`: We defined custom flags for Energy Attack. Currently, this argument should be used together with Energy Attack only.
  - `ea:annealoff`: Whether to enable annealing of Energy Attack. **Always add this flag when using Energy Attack for better performance.**
  - `ea:basepkl:<your_pkl>`: Changes the perturbation patches used for Energy Attack. By default we use patches from an adversarially-trained CIFAR10 ResNet18. For details please refer to [this section](#using-other-perturbation-patches).

### Using other perturbation patches

The `--custom_flags` for Energy Attack allows users to specify a certain set of perturbation patches. We provide two sets of patches

- `model1_fw_5-pca.pkl`: Perturbation patches from a (non-robust) CIFAR10 ResNet34.
- `model2_fw_5-pca.pkl`: Perturbation patches from a robust (trained with AT) CIFAR10 ResNet18.

To use a certain patch, you can append argument like:

```shell
ea:basepkl:model1_fw_5-pca.pkl
```
