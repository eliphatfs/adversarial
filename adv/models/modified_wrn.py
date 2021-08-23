from torch import nn
from .wide_resnet import WideResNet


class ModifiedWRN28(WideResNet):
    def __init__(self):
        super().__init__(28)
        self.fc = nn.Sequential(
            nn.Linear(640, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )
