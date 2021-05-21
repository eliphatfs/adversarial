import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from VAEgis import VAEgisMkII, load_model


def VAEgis_validation(
        model: VAEgisMkII,
        valid_loader: DataLoader,):
    avg_err = 0
    avg_acc = 0
    model.train(False)
    valid_progress = tqdm(valid_loader)
    for i, (x, y) in enumerate(valid_progress):
        x = x.cuda()
        y = y.cuda()
        pred = model.forward(x)
        err = F.cross_entropy(pred, y)
        avg_err += err.item()
        result: torch.Tensor = pred.argmax(-1) == y
        avg_acc += result.float().mean().item()
        valid_progress.set_description(
            "Loss: {:.4f} | Acc: {:.4f}".format(avg_err/(i+1), avg_acc/(i+1)))
        valid_progress.refresh()
    print("[Validation Result] Loss: {:.4f} | Acc: {:4f}".format(
        avg_err/(i+1), (avg_acc/(i+1))),
        file=sys.stderr,)
    model.train(True)


def VAEgis_train(
        model: VAEgisMkII,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        n_epoch=500,
        checkpoint_interval=5,):
    for e in range(n_epoch):
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
                "Epoch: {:d} | Loss: {:.4f} | Acc: {:.4f}".format(
                    e+1, avg_err/(i+1), avg_acc/(i+1)))
            train_progress.refresh()

        if e % checkpoint_interval == 0:
            print("Checkpoint!", file=sys.stderr)
            state_dict = model.state_dict()
            torch.save(state_dict, './vaegis.pth')
            VAEgis_validation(model, valid_loader)


if __name__ == "__main__":
    MAX_EPOCH = 100

    encoder, [train_loader, valid_loader, test_loader] = load_model('CIFAR10')
    model = VAEgisMkII(encoder)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters())

    model.train()
    VAEgis_train(model, optimizer, train_loader, valid_loader)
