from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models import resnet18
from models import inception_v3
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


class Cifar10Renormalize(torch.nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x):
        x = x - x.new_tensor([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        x = x / x.new_tensor([0.2023, 0.1994, 0.201]).reshape(1, 3, 1, 1)
        return self.wrap(x)


class ImageNetRenormalize(torch.nn.Module):
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap
        self.sizer = transforms.Resize(
            384, interpolation=InterpolationMode.BICUBIC)

    def forward(self, x):
        x = (x - 0.5) / 0.5
        return self.wrap(self.sizer(x))


def get_model_for_attack(model_name):
    if model_name == 'model_vgg16bn':
        model = vgg16_bn(pretrained=True)
    elif model_name == 'model_resnet18':
        model = resnet18(pretrained=True)
    elif model_name == 'model_inceptionv3':
        model = inception_v3(pretrained=True)
    elif model_name == 'model_vitb':
        from mnist_vit import ViT, MegaSizer
        model = MegaSizer(ImageNetRenormalize(
            ViT('B_16_imagenet1k', pretrained=True)))
    elif model_name.startswith('model_hub:'):
        _, a, b = model_name.split(":")
        model = torch.hub.load(a, b, pretrained=True)
        model = Cifar10Renormalize(model)
    elif model_name.startswith('model_mnist:'):
        _, a = model_name.split(":")
        model = torch.load('mnist.pt')[a]
    elif model_name.startswith('model_ex:'):
        _, a = model_name.split(":")
        model = torch.load(a)
    else:
        raise ValueError(f'Model f{model_name} does not exist.')
    return model
