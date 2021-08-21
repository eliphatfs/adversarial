from utils import get_dataset_size
import torch
import pickle

from attack.pgd_attack import pgd_attack
from tqdm import trange


def eval_model(model, test_loader, device, dataset):
    correct = []
    distance = []
    num = 0
    with trange(get_dataset_size(dataset)) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            model.eval()
            with torch.no_grad():
                output = model(x)
            pred = output.argmax(dim=1)
            correct.append(pred == label)
            num += x.shape[0]
            pbar.set_description(
                f"Acc: {torch.cat(correct).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    return natural_acc, distance


def eval_model_pgd(
        model, test_loader, device, step_size, epsilon, perturb_steps, dataset):
    correct_adv, correct = [], []
    distance = []
    num = 0
    with trange(get_dataset_size(dataset)) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            x_adv = pgd_attack(model, x.clone(), label.clone(),
                               step_size, epsilon, perturb_steps)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(
                torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            num += x.shape[0]
            pbar.set_description(
                f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    return natural_acc, robust_acc, distance


def eval_model_with_attack(
        model, test_loader, attack, epsilon, device, dataset):
    orig_pic, adv_pic, perturb = [], [], []
    correct_adv, correct = [], []
    distance = []
    num = 0
    with trange(get_dataset_size(dataset)) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            x_adv = attack(model, x.clone(), label.clone())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = x_adv.clamp(0, 1)
            batch_orig_pic = x.cpu().detach().numpy()
            batch_adv_pic = x_adv.cpu().detach().numpy()
            orig_pic.extend(batch_orig_pic)
            adv_pic.extend(batch_adv_pic)
            perturb.extend(batch_adv_pic - batch_orig_pic)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(
                torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            num += x.shape[0]
            pbar.set_description(
                f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    distance = torch.cat(distance).max()
    # pickle.dump(orig_pic, open('./pics_ori_mdl3_fw.pkl', 'wb'))
    # pickle.dump(adv_pic, open('./pics_adv_mdl3_fw.pkl', 'wb'))
    pickle.dump(perturb, open('./pics_ptb_vgg16bn_fw.pkl', 'wb'))
    return natural_acc, robust_acc, distance


def eval_model_with_targeted_attack(
        model, test_loader, attack, epsilon, device, dataset):
    correct_adv, correct = [], []
    succeeded_adv = []
    distance = []
    num = 0
    with trange(get_dataset_size(dataset)) as pbar:
        for x, label in test_loader:
            x, label = x.to(device), label.to(device)
            batch, c, h, w = x.shape
            y_tgt = torch.ones_like(label).to(device)
            x_adv = attack.targeted(model, x.clone(), y_tgt.clone())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = x_adv.clamp(0, 1)
            model.eval()
            with torch.no_grad():
                output = model(x)
                output_adv = model(x_adv)
            distance.append(
                torch.max((x-x_adv).reshape(batch, -1).abs(), dim=1)[0])
            pred = output.argmax(dim=1)
            pred_adv = output_adv.argmax(dim=1)
            correct.append(pred == label)
            correct_adv.append(pred_adv == label)
            succeeded_adv.append(y_tgt == pred_adv)
            num += x.shape[0]
            pbar.set_description(
                f"Acc: {torch.cat(correct).float().mean():.5f}, Robust Acc:{torch.cat(correct_adv).float().mean():.5f}, Targeted Success Rate: {torch.cat(succeeded_adv).float().mean():.5f}")
            pbar.update(x.shape[0])
    natural_acc = torch.cat(correct).float().mean()
    robust_acc = torch.cat(correct_adv).float().mean()
    success_rate = torch.cat(succeeded_adv).float().mean()
    distance = torch.cat(distance).max()
    return natural_acc, robust_acc, success_rate, distance
