import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import src


def load_model(dataset, device='cuda'):
    train_loader, valid_loader, test_loader = src.dataloader(dataset)
    data_shape = src.utils.get_data_shape(train_loader)

    pth = '.\\src\\models'
    pth = os.path.join(pth, 'pretrained', 'VAE', 'CIFAR10')
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    model = src.VAE(data_shape).to(device)
    model.initialize(train_loader)

    checkpoint = torch.load(pth_train)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model successfully loaded!')
    except RuntimeError as err:
        print('? Fucked up loading model.')
        print(err.__doc__)
        quit()

    return model, [train_loader, valid_loader, test_loader]


class VAEgis(nn.Module):
    def __init__(self, encoder: src.VAE):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512, 10)

    def forward(self, x) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                sample = self.encoder.encode(x)
            out = self.fc(sample.reshape(-1, 512))
            return out
        else:
            return self.fc(self.encoder.get_mean(x))


class VAEgisMkII(nn.Module):
    def __init__(self, encoder: src.VAE):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 648)
        self.fc3 = nn.Linear(648, 10)

    def forward(self, x) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                sample = self.encoder.encode(x)
            out = self.fc1(sample.reshape(-1, 512))
            out = F.sigmoid(out)
            out = self.fc2(out)
            out = F.sigmoid(out)
            out = self.fc3(out)
            return out


if __name__ == "__main__":
    load_model('CIFAR10')
