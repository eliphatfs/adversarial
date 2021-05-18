import torch
import torch.nn as nn
import tqdm
from utils import prepare_cifar


device = 'cuda:0'
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.AvgPool2d(2),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.AvgPool2d(2),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.AvgPool2d(2),
    nn.Conv2d(256, 512, kernel_size=3, padding=1),
    nn.Conv2d(512, 512, kernel_size=3, padding=1),
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(512 * 4 * 4, 10)
).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
train_loader, test_loader = prepare_cifar(32, 64)
opt = torch.optim.Adam(model.parameters(), weight_decay=0.000)
print("Initialized.")


for epoch in range(50):
    with tqdm.tqdm(train_loader, leave=True) as train:
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            acc = (pred.argmax(-1) == y).float().mean().item()
            running_acc += acc
            running_loss += loss.item()
            train.set_description("CL %.4f, Acc %.4f" % (
                running_loss / (i + 1), running_acc / (i + 1)
            ))
    with tqdm.tqdm(test_loader, leave=True) as test:
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(test):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)
                acc = (pred.argmax(-1) == y).float().mean().item()
            running_acc += acc
            running_loss += loss.item()
            test.set_description("CL %.4f, Acc %.4f" % (
                running_loss / (i + 1), running_acc / (i + 1)
            ))
