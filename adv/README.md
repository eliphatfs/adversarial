# AI2612 Machine Learning Project

## Baseline is all you need

- This is the repository for SJTU AI2612 Course Project \#2: Adversarial attacks and defenses.
- Course Project Repository: <https://github.com/jiangyangzhou/defense_base>

## Table of Contents

- [How To Run](#how-to-run)
  - [Loading Pretrained Models](#pretrained-models)
  - [Attack](#attack)
  - [Defense](#defense)
  - [Other](#other-executable-code)
- [Environment](#environment)

## How to Run

### Pretrained Models

- Weight of models should be put under directory `./models/weights`
- We provide two pre-trained Wide ResNet 28 models.
  - `WRN28_FWAWP_TRADES`: Wide ResNet 28 trained with FW-AdAmp, AWP and TRADES
  - `WRN28_FWAWP`: Wide ResNet 28 trained with FW-AdAmp and AWP
- Other models will be made available soon on jbox (TODO)
- **NOTE: The loaded files are [checkpoints](#checkpoint) and contains more than the state dict of the model, so remember to use `model` to access the actual state dict of the model**

### Attack

#### FW-AdAmp

To evaluate our trained model, please run:

```shell
python3 attack_main.py --attacker fw --model_name WRN28_FWAWP
```

or

```shell
python3 attack_main.py --attacker fw --model_name WRN28_FWAWP_TRADES
```

To run a general FW-AdAmp attack, please execute the following:

```shell
python3 attack_main.py --attacker fw --model_name <model_name>
```

`<model_name>` can be one of:

- Models provided by TA
  - `model1`: Basic ResNet 34.
  - `model2`: ResNet 18 w/ AT.
  - `model3`: Small ResNet w/ gradient perturbation.
  - `model4`: Wide ResNet w/ TRADES loss.
  - `model5`: Wide ResNet w/ Hypersphere Embedding.
  - `model6`: Wide ResNet w/ AT-AWP.
- Models trained by us (see [pretrained models](#pretrained-models)):
  - `WRN28_FWAWP_TRADES`: included in submission
  - `WRN28_FWAWP`: included in submission
  - `PreActRN18_FWAWP`: NOT included in submission. Pre-activate ResNet 18 trained with FW-AdAmp w/ AWP
  - `PreActRN34_FWAWP`: NOT included in submission. Pre-activate ResNet 34 trained with FW-AdAmp w/ AWP
  - `RN34_FWAWP`: NOT included in submission. ResNet 34 trained with FW-AdAmp w/ AWP
  - `RN18_FWAWP`: NOT included in submission. ResNet 18 trained with FW-AdAmp w/ AWP
  - `RN18_FWAWP_TRADES`: NOT included in submission. ResNet 18 trained with FW-AdAmp w/ AWP and TRADES loss
  - `WRN28_ATAWP`: NOT included in submission. Wide ResNet 34 trained with AT-AWP
  - `WRN28_ATAWP_TRADES`: NOT included in submission. Wide ResNet 34 trained with TRADES-AWP
- If no model name is provided (or `""` is provided), then the program will attempt to load a **checkpoint** from a path specified by `--model_path`. In this case the argument `--model` must be given to specify the architecture of the model.

#### Other attacks

`attack_main.py` also supports other attacks. To use alternative attacks, change parameter of `--attacker` into one of the following:

- `pgd`: Baseline PGD attack provided by TA.
- `arch_transfer`: Arch Transfer Attack. **NOTE: this attack somehow does not run on pytorch 1.8.1+cu102. But it should run on pytorch 1.7.1+cu101.**
- `barrier`: Barrier Method Attack, solves the attacking optimization with Barrier Method.
- `stochastic_sample`: Basic random sample attack.
- `sobol_sample`: Improved random sampling attack, uses Sobol sequence for sampling.
- `deepfool`: Deep Fool attack.
- `second_order`: Solves the attacking optimization with second-order optimization methods.

### Defense

#### Train a New Model

To train a robust model with FW-AdAmp, please execute the following:

```shell
python3 fw_awp_train.py --attacker fw --model WideResNet28 --awp_warmup 10 --trades
```

- This command trains a Wide ResNet 28 with Adversarial Weight Perturbation and TRADES loss.
- In addition to `fw`, we also support `--attacker pgd` for standard AT.
- In addition to `WideResNet28`, we also support other models for `--model`
  - `ResNet18`
  - `ResNet34`
  - `PreActResNet18`
  - `PreActResNet34`
  - Pre-activate ResNet models are copied directly from <https://github.com/csdongxian/AWP/tree/main/AT_AWP>

#### Checkpoint Saving

Models by default will be saved to `./logs/<start_time><name>`

- `<start_time>` is the time when the training started.
- `<name>` can be specified by `--model_name` argument. Default name is `baseline`.

In that directory there will be

- `train.log`: log file of the training.
- `<name>-final.pt`: The **final** weight for the model.
- `<name>-best.pt`: The **checkpoint** for the model with **the best robust acc**.
- `<name>-checkpoint.pt`: The **checkpoint** for the model in **the latest epoch**.
- `<name>-footprints.pkl`: A pickle binary file recording the loss, accuracy and robust accuracy of the model.

##### Checkpoint

The **checkpoint** is a python dictionary

```python
{
    'epoch': e,
    'model': model.module.state_dict(),
    'model_optim': model_optimizer.state_dict(),
    'proxy_optim': proxy_optimizer.state_dict(),
}
```

It records the epoch of current training process, the state dictionary of the model and the optimizer, and the state dictionary of the proxy optimizer (proxy is used in AWP).

#### Resume a Training Process

To resume training please run the following:

```shell
python3 fw_awp_train.py --attacker fw --model WideResNet28 --awp_warmup 10 --resume --checkpoint_path <path_to_checkpoint>
```

Notice that a `--checkpoint_path` argument must be supplied. It should specify the path to a saved [checkpoint](#checkpoint) of a pre-trained model.

### Other Executable Code

These files are not directly related with the main part (attacks and defenses) of our project. However, since they still play an important role in the entire processs of accomplishing this course project, we breifly document their usage here.

- `adamp_stats.py` and `visualization.py` are used for drawing fancy graphics and are not designed to be runned for experimental purposes.
- `aegleseeker.py` implements a Lagrangian regularization on convolution kernels. This method is named after [Aegleseeker](https://arcaea.fandom.com/wiki/Aegleseeker).
- `experimental.py` is some attempts on using a KNN classifier.
- `krylov_analysis.py` :dove: Please refer to our report.
- `niyf.py` Noise Is Your Friend. Based on the idea of reducing variance by sampling multiple times with additive noise. ~~But it does not work as desired~~.
- `prover.py` :dove: Please refer to our report.
- `radam.py` An optimizer. For details please refer to [this repository](https://github.com/LiyuanLucasLiu/RAdam)
- `reg_mod.py` Our struggle to find proper regularization.
- `vae.py` It is not encouraged to run this file because it is currently not related with our project at all. The story behind this Variational AutoEncoder is currently pigeoned. Stay tuned. This file is currently not related with our project. For more details, please refer to [this](https://github.com/eliphatfs/adversarial/blob/main/Report/bullshitting.tex).

## Environment

All the code, unless otherwise stated, should run properly with

- pytorch 1.8.1 with cuda 10.2 or pytorch 1.7.1 with cuda 10.1
- numpy 1.19.2
- scipy 1.6.2

## Misc

### Development Logs

[DevLog](./DevLog.md)
