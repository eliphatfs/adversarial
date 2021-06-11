# AI2612 Machine Learning Project

## Baseline is all you need

- This is the repository for SJTU AI2612 Course Project \#2: Adversarial attacks and defenses.
- Course Project Repository: <https://github.com/jiangyangzhou/defense_base>

## Table of Contents

- [How To Run](#how-to-run)
  - [Attack](#attack)
  - [Defense](#defense)

## How to Run

### Attack

#### FW-AdAmp

To run FW-AdAmp attack, please execute the following:

```shell
python3 attack_main.py --attacker fw --model <model_name>
```

`<model_name>` can be one of:

- Models provided by TA
  - `model1`: Basic ResNet 34.
  - `model2`: ResNet 18 w/ AT.
  - `model3`: Small ResNet w/ gradient perturbation.
  - `model4`: Wide ResNet w/ TRADES loss.
  - `model5`: Wide ResNet w/ Hypersphere Embedding.
  - `model6`: Wide ResNet w/ AT-AWP.
- Models trained by us:

#### Other attacks

`attack_main.py` also supports other attacks. To use alternative attacks, change parameter of `--attacker` into one of the following:

- `pgd`: Baseline PGD attack provided by TA.
- `arch_transfer`: Arch Transfer Attack.
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
- `<name>` can be specified by `--model_name` argument. Default name is `myModel`.

In that directory there will be

- `train.log`: log file of the training.
- `<name>-final.pt`: The **final** weight for the model.
- `<name>-best.pt`: The **checkpoint** for the model with **the best robust acc**.
- `<name>-checkpoint.pt`: The **checkpoint** for the model in **the latest epoch**.
- `<name>-footprints.pkl`: A pickle binary file recording the loss, accuracy and robust accuracy of the model.

##### Checkpoint

The checkpoint is a python dictionary

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

### Other Code

These files are not directly related with the main part (attacks and defenses) of our project. However, since they still play an important role in the entire processs of accomplishing this course project, we breifly document their usage here.

## Environment

## Misc

### Development Logs

[DevLog](./DevLog.md)
