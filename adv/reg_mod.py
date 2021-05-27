import torch
import torch.nn.functional as F
from utils import prepare_cifar
import tqdm
import radam
import vgg


device = 'cuda:1'
model = vgg.vgg13_bn().to(device)
train_loader, test_loader = prepare_cifar(200, 400)
optim = radam.RAdam(model.parameters())
epsilon = 8 / 255


for epoch in range(240):
    with tqdm.tqdm(train_loader) as train:
        running_loss = 0.0
        running_grad = 0.0
        running_acc = 0.0
        model.train()
        for i, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            x_rg = x.detach().clone().requires_grad_()
            pred = model(x_rg)
            loss = F.cross_entropy(pred, y)
            # act_norm = model.normacc.value_and_clear()
            xg = torch.autograd.grad(loss, x_rg)[0]
            xgf = xg.flatten(1)
            pred_at = model(x + xg / (1e-7 + torch.norm(xgf, dim=-1).reshape(-1, 1, 1, 1)) * epsilon * xgf.shape[-1] ** 0.5)
            loss_at = F.cross_entropy(pred_at, y)
            loss_at.backward()
            running_grad += (pred_at.argmax(-1) == y).float().mean().item()
            optim.step()
            running_loss += loss_at.item()
            running_acc += (pred.argmax(-1) == y).float().mean().item()
            train.set_description("Loss: %.4f, FAcc: %.4f, Acc: %.4f" %
                (running_loss / (i + 1), running_grad / (i + 1), running_acc / (i + 1))
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
