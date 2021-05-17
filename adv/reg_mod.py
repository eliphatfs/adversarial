import torch
from utils import prepare_cifar
import tqdm
import vgg


device = 'cuda:1'
model = vgg.vgg13_bn().to(device)
train_loader, test_loader = prepare_cifar(100, 128)
optim = torch.optim.Adam(model.parameters())


for epoch in range(50):
    for g in optim.param_groups:
        g['weight_decay'] = 0.1 * epoch / 50
    with tqdm.tqdm(train_loader) as train:
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.cross_entropy(pred, y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            running_acc += (pred.argmax(-1) == y).float().mean().item()
            train.set_description("Loss: %.4f, Acc: %.4f" %
                (running_loss / (i + 1), running_acc / (i + 1))
            )
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = torch.nn.functional.cross_entropy(pred, y)
                running_loss += loss.item()
                running_acc += (pred.argmax(-1) == y).float().mean().item()
        print("Epoch: %d, Loss: %.4f, Acc: %.4f" %
            (epoch, running_loss / (i + 1), running_acc / (i + 1))
        )
        torch.save(model.state_dict(), "vgg13bn_regm.dat")