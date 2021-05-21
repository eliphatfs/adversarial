import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from VAEgis import VAEgisMkII, load_model

if __name__ == "__main__":
    MAX_EPOCH = 100

    encoder, [train_loader, valid_loader, test_loader] = load_model('CIFAR10')
    model = VAEgisMkII(encoder)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for e in range(MAX_EPOCH):
        train_progress = tqdm(train_loader)
        avg_err = 0
        avg_acc = 0
        for i, (x, y) in enumerate(train_progress):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            pred = model.forward(x)
            err = F.cross_entropy(pred, y)
            err.backward()
            optimizer.step()
            avg_err += err.item()
            result: torch.Tensor = pred.argmax(-1) == y
            avg_acc += result.float().mean().item()
            train_progress.set_description(
                "Loss: {:.4f} | Acc: {:.4f}".format(avg_err/(i+1), avg_acc/(i+1)))
