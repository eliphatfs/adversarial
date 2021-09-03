from utils import get_test_imagenet
import cv2
import torch
import numpy as np
import torch.nn.functional as F


args = {
    'number_images': 1000,
    'model': 'model_inception',  # model to be attacked
    'method': 'uniform',  # or biased, average, fixed biased, fixed average
    'norm': 'linf',  # linf only
    'use_larger_step_size': True,
    'input_dir': 'ultimately_working',
    'img_size': 299,
    'device': 'cuda:0',
    'max_queries': 10000,
    'grad_debug': True,
    'samples_per_draw': 10,
    'dataprior': True,
    'fixed_const': 0
}


def get_1em12_tensor(device):
    return torch.Tensor([1e-12]).to(device)


def safe_sqrt_mean_square(x, device):
    return torch.maximum(
        get_1em12_tensor(device),
        torch.sqrt(torch.mean(torch.square(x))))


def get_imagenet_models(model_name):
    if model_name == 'model_vgg16bn':
        from models import vgg16_bn
        model = vgg16_bn(pretrained=True)
    elif model_name == 'model_resnet18_imgnet':
        from models import resnet18
        model = resnet18(pretrained=True)
    elif model_name == 'model_inception':
        from models import inception_v3
        model = inception_v3(pretrained=True)
    else:
        raise ValueError(f'Buggya no model named {model_name}')
    print(f'Model: {model_name}')
    return model


model = get_imagenet_models(args['model']).to(args['device'])
model.eval()

if args['method'] != 'uniform':
    from models.resnet_imgnt import resnet152
    model_s = resnet152(pretrained=True)
    model_s.eval()


# we only use hyper params of linf setting
# check original repo for hyperparams of l2 distance
epsilon = 0.05
eps = epsilon
learning_rate = 0.005
ini_sigma = 1e-4

success = 0
queries = []
correct = 0

img_counter = 0

test_loader = get_test_imagenet(1)
for image, label in test_loader:
    img_counter += 1
    image = image.to(args['device'])
    label = label.to(args['device'])
    sigma = ini_sigma
    np.random.seed(0)
    torch.random.manual_seed(0)

    adv_image = image.detach().clone()
    ori_image = image.detach().clone()
    ori_pred_prob = model(ori_image)
    ori_pred = ori_pred_prob.argmax(dim=1)
    loss = F.cross_entropy(ori_pred_prob, label)

    lr = learning_rate
    losses = []
    total_q = 0
    ite = 0
    device = args['device']

    while total_q <= args['max_queries']:
        total_q += 1

        if ite % 2 == 0 and sigma != ini_sigma:
            print('sigma has been modified before')
            print('Checking if sigma could be reset to ini_sigma')
            rand = torch.randn_like(adv_image).to(device)
            rand = rand / safe_sqrt_mean_square(rand, device)

            rand_pred = model(adv_image + ini_sigma * rand)
            rand_loss = F.cross_entropy(rand_pred, label)
            total_q += 1

            rand = torch.randn_like(adv_image).to(device)
            rand = rand / safe_sqrt_mean_square(rand, device)

            rand_pred = model(adv_image + ini_sigma * rand, label)
            rand_loss2 = F.cross_entropy(rand_pred, label)
            total_q += 1

            if (rand_loss - loss)[0] != 0 and (rand_loss2 - loss)[0] != 0:
                print("set sigma back to ini_sigma")
                sigma = ini_sigma

        if args['method'] != 'uniform':
            adv_image.requires_grad_()
            with torch.enable_grad():
                loss_s = F.cross_entropy(model_s(adv_image), label)
                prior = torch.autograd.grad(loss_s, [adv_image])[0]
                adv_image.requires_grad_(False)

            prior = prior / safe_sqrt_mean_square(prior, device)

        if args['method'] in ['biased', 'average']:
            start_iter = 3
            if ite % 10 == 0 or ite == start_iter:
                s = 10
                pert = torch.randn(size=(s,) + adv_image.shape[1:]).to(device)
                for i in range(s):
                    pert[i] = pert[i] / safe_sqrt_mean_square(pert[i], device)
                eval_points = adv_image.detach().clone() + sigma * pert
                losses = F.cross_entropy(
                    model(eval_points),
                    torch.full((s,), label.item()).to(device))
                total_q += s
                norm_square = torch.mean(((losses - loss) / sigma)**2)

            while True:
                prior_loss = F.cross_entropy(
                    model(adv_image.detach().clone() + sigma * prior),
                    label)
                total_q += 1
                diff_prior = (prior_loss - loss)[0]
                if diff_prior == 0:
                    sigma *= 2
                    print('Multiplied sigma by 2')
                else:
                    break

            est_alpha = diff_prior / sigma / \
                torch.maximum(
                    get_1em12_tensor(device),
                    torch.sqrt(torch.sum(torch.square(prior))*norm_square))
            print('Estimated alpha:', est_alpha)
            alpha = est_alpha
            if alpha < 0:
                prior = -prior
                alpha = -alpha

        q = args['samples_per_draw']
        n = args['img_size'] * args['img_size'] * 3
        d = 50 * 50 * 3
        gamma = 3.5
        A_square = d / n * gamma

        return_prior = False
        if args['method'] == 'average':
            if args['dataprior']:
                alpha_nes = torch.sqrt(A_square * q / (d + q - 1))
            else:
                alpha_nes = torch.sqrt(q / (n + q - 1))
            if alpha >= 1.414 * alpha_nes:
                return_prior = True
        elif args['method'] == 'biased':
            if args['dataprior']:
                best_lambda = A_square * (A_square - alpha ** 2 * (d + 2 * q - 2)) / (
                    A_square ** 2 + alpha ** 4 * d ** 2 - 2 * A_square * alpha ** 2 * (q + d * q - 1))
            else:
                best_lambda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * q - 2)) / (
                    alpha ** 4 * n * (n + 2 * q - 2) - 2 * alpha ** 2 * n * q + 1)
            print('best_lambda = ', best_lambda)
            if best_lambda < 1 and best_lambda > 0:
                lmda = best_lambda
            else:
                if alpha ** 2 * (n + 2 * q - 2) < 1:
                    lmda = 0
                else:
                    lmda = 1
            if torch.abs(alpha) >= 1:
                lmda = 1
            print('lambda = ', lmda)
            if lmda == 1:
                return_prior = True
        elif args['method'] == 'fixed_biased':
            lmda = args['fixed_const']

        if not return_prior:
            if args['dataprior']:
                pert = np.random.normal(size=(q, 50, 50, 3))
                pert = np.array(
                    [
                        cv2.resize(
                            pert[i], adv_image.shape[2:],
                            interpolation=cv2.INTER_NEAREST)
                        for i in range(q)
                    ])
                pert = np.transpose(pert, (0, 3, 1, 2))
                pert = torch.Tensor(pert).to(device)
            else:
                pert = torch.randn(size=(q,) + adv_image.shape[1:]).to(device)
            for i in range(q):
                if args['method'] in ['biased', 'fixed_biased']:
                    pert[i] = pert[i] - \
                        torch.sum(pert[i] * prior) * prior /\
                        torch.maximum(
                            get_1em12_tensor(device),
                            torch.sum(prior*prior))
                    pert[i] = pert[i] /\
                        safe_sqrt_mean_square(pert[i], device)
                    pert[i] = torch.sqrt(1-lmda) * \
                        pert[i] + torch.sqrt(lmda) * prior
                else:
                    pert[i] = pert[i] / safe_sqrt_mean_square(pert[i], device)

            while True:
                eval_points = adv_image + sigma * pert
                losses = F.cross_entropy(
                    model(eval_points), torch.full((q,), label.item()).to(device))
                total_q += q

                grad = (losses - loss).reshape(-1, 1, 1, 1) * pert
                grad = torch.mean(grad, axis=0)
                norm_grad = torch.sqrt(torch.mean(torch.square(grad)))
                if norm_grad == 0:
                    sigma *= 5
                    print('Estimated grad is 0, multiply sigma by 5')
                else:
                    break
            grad = grad / safe_sqrt_mean_square(grad, device)

            if args['method'] == 'average':
                while True:
                    diff_pred = model(
                        adv_image.detach().clone() + sigma * prior)
                    diff_prior = (F.cross_entropy(diff_pred, label) - loss)[0]
                    total_q += 1

                    diff_pred = model(
                        adv_image.detach().clone() + sigma * grad)
                    diff_nes = (F.cross_entropy(diff_pred, label) - loss)[0]
                    total_q += 1
                    diff_prior = max(0, diff_prior)
                    if diff_prior == 0 and diff_nes == 0:
                        sigma *= 2
                        print('Multiplied sigma by 2')
                    else:
                        break
                final = prior * diff_prior + grad * diff_nes
                final = final / safe_sqrt_mean_square(final, device)
                print(f'diff_prior = {diff_prior}, diff_nes = {diff_nes}')
            elif args['method'] == 'fixed_average':
                diff_pred = model(
                    adv_image.detach().clone() + sigma * prior)
                diff_prior = (F.cross_entropy(diff_pred, label) - loss)[0]
                total_q += 1
                if diff_prior < 0:
                    prior = -prior
                final = args['fixed_const'] * prior +\
                    (1 - args['fixed_const']) * grad
                final = final / safe_sqrt_mean_square(final, device)

            else:
                final = grad

        else:
            final = prior

        # only consider linf case
        adv_image = adv_image + lr * torch.sign(final)
        adv_image = torch.min(torch.max(adv_image, image - epsilon), image + epsilon)
        adv_image = adv_image.clamp(0, 1)

        adv_prob = model(adv_image)
        adv_label = adv_prob.argmax(dim=1)
        adv_loss = F.cross_entropy(adv_prob, label)

        print(
            'Image:', img_counter,
            'Queries:', total_q,
            'Loss:', adv_loss.item(),
            'Prediction:', adv_label.item(),
            'OriginalPred:', ori_pred.item(),
            'Truth:', label.item(), '\r', flush=True)
        ite += 1

        if adv_label != label:
            print('Done at query:', total_q)
            success += 1
            queries.append(total_q)
            break

print('Success Rate:', success / args['number_images'])
