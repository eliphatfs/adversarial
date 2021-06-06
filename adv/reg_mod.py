import torch
import torch.nn.functional as F
from utils import prepare_cifar
import tqdm
import radam
from models import PreActResNet18
from aegleseeker import AegleSeeker
from eval_model import eval_model_pgd


device = 'cuda:0'
model = PreActResNet18()
model = AegleSeeker(model).to(device)
train_loader, test_loader = prepare_cifar(100, 100)
optim = radam.RAdam(model.parameters())
epsilon = 8 / 255


for epoch in range(34):
    with tqdm.tqdm(train_loader) as train:
        running_loss = 0.0
        running_grad = 0.0
        running_acc = 0.0
        model.train()
        for i, (x, y) in enumerate(train):
            x, y = x.to(device), y.to(device)
            for _ in range(3):
                x_rg = x.detach().clone().requires_grad_(True) + torch.randn_like(x) * epsilon / 2
                optim.zero_grad()
                pred = model(x_rg)
                loss = F.cross_entropy(pred, y)
                L = [0.3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                # AdAmp (maybe not a good name for the op)
                loss_amp = sum(w ** 2 * F.cross_entropy(pred * w, y) for w in L)
                grad = torch.autograd.grad(loss_amp, [x_rg], create_graph=True)[0]
                reg = torch.norm(grad.flatten(1), dim=-1).mean()
                # reg = model.reg()
                (loss + 0.09 * reg).backward()
                x = x.detach().clone() + torch.sign(grad.detach().clone()) * epsilon
                optim.step()
                running_loss += loss.item()
                running_grad += reg.item()
                running_acc += (pred.argmax(-1) == y).float().mean().item()
                train.set_description("Loss: %.4f, Reg: %.4f, Acc: %.4f" %
                    (running_loss / (i + 1), running_grad / (i + 1), running_acc / (i + 1))
                )
        model.eval()
        if epoch % 1 == 0:
            test_acc, test_robust_acc, _ = eval_model_pgd(
                model, test_loader, device,
                0.007, 8 / 255, 10
            )
            print("Epoch: %d, Robust: %.4f, Acc: %.4f" %
                (epoch, test_robust_acc, test_acc)
            )
        torch.save(model.state_dict(), "vgg13bn_regm.dat")
