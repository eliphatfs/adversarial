from torch import nn
from torchvision import transforms
import torch
import tqdm
import numpy
from utils import get_test_mnist


if __name__ == '__main__':
    dev = 'cpu'
    cnn = nn.Sequential(
        nn.Conv2d(3, 12, 5),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(12, 36, 5),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 36, 10)
    ).to(dev)
    trs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x] * 3))
    ])
    ys = []
    with torch.no_grad():
        for x, y in tqdm.tqdm(get_test_mnist(100)):
            x, y = x.to(dev), y.to(dev)
            logit = cnn(x).cpu().numpy()
            ys.extend(logit.argmax(-1))
    torch.save(cnn, "exp_rand_mnist_cnn.pt")
    numpy.save("rand_mnist.npy", ys)
