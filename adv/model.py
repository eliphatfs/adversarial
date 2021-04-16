from models import WideResNet, ResNet18
from models import ResNet34, SmallResNet, WideResNet28, WideResNet34
import torch


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_w(model, path):
    pref = next(model.parameters())
    model.load_state_dict(torch.load(path, map_location=pref.device))


def get_model_for_attack(model_name):
    if model_name == 'model1':
        model = ResNet34()
        load_w(model, "./models/weights/resnet34.pt")
    elif model_name == 'model2':
        model = ResNet18()
        load_w(model, "./models/weights/resnet18_AT.pt")
    elif model_name == 'model3':
        model = SmallResNet()
        load_w(model, "./models/weights/res_small.pth")
    elif model_name == 'model4':
        model = WideResNet34()
        pref = next(model.parameters())
        model.load_state_dict(filter_state_dict(torch.load(
            "./models/weights/trades_wide_resnet.pt",
            map_location=pref.device
        )))
    elif model_name == 'model5':
        model = WideResNet()
        load_w(model, "./models/weights/wideres34-10-pgdHE.pt")
    elif model_name == 'model6':
        model = WideResNet28()
        pref = next(model.parameters())
        model.load_state_dict(filter_state_dict(torch.load(
            'models/weights/RST-AWP_cifar10_linf_wrn28-10.pt',
            map_location=pref.device
        )))
    return model
