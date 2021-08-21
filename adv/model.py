from models import resnet18
from models import inception_v3
from models import WideResNet, ResNet18
from models import SmallResNet, WideResNet28, WideResNet34
from models import PreActResNet18, PreActResNet34
from models import ResNet34
from models import vgg16_bn
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


def get_model_for_defense(model_name):
    if 'WRN28' in model_name:
        model = WideResNet28()
    elif 'PreActRN18' in model_name:
        model = PreActResNet18()
    elif 'PreActRN34' in model_name:
        model = PreActResNet34()
    elif 'RN18' in model_name:
        model = ResNet18()
    elif 'RN34' in model_name:
        model = ResNet34()
    else:
        raise ValueError(
            f'Unsupported model name: {model_name}. Check your spelling!')
    checkpoint = torch.load(f"./models/weights/{model_name}.pt")
    model.load_state_dict(checkpoint['model'])

    return model


def get_custom_model(model, model_path):
    if model == '':
        raise ValueError('Please specify a model architecture!')
    if model_path == '':
        raise ValueError('Please specify a path to checkpoint!')

    model = _get_specified_model(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    return model


def _get_specified_model(model):
    if model == 'ResNet18':
        return ResNet18()
    elif model == 'ResNet34':
        return ResNet34()
    elif model == 'PreActResNet18':
        return PreActResNet18()
    elif model == 'PreActResNet34':
        return PreActResNet34()
    elif model == 'WideResNet28':
        return WideResNet28()
    elif model == 'WideResNet34':
        return WideResNet34()
    else:
        return ResNet18()
    # fixme: remove support for WRN34


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
    elif model_name == 'model_vgg16bn':
        model = vgg16_bn(pretrained=True)
    elif model_name == 'model_resnet18_imgnet':
        model = resnet18(pretrained=True)
    elif model_name == 'model_inception':
        model = inception_v3(pretrained=True)
    elif model_name.startswith('model_hub:'):
        _, a, b = model_name.split(":")
        model = torch.hub.load(a, b, pretrained=True)
    return model
