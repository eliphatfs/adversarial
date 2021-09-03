from torch import nn
from .wide_resnet import WideResNet
from .resnet import ResNet, BasicBlock


class ModifiedWRN28(WideResNet):
    def __init__(self):
        super().__init__(28)
        self.fc = nn.Sequential(
            nn.Linear(640, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

class ModifiedResNet18(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.linear = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 3000),
            nn.ReLU(),
            nn.Linear(3000, 10)
        )
