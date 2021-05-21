""" PyTorch implimentation of VAE and Super-Resolution VAE.

    Reposetory Author:
        Ioannis Gatopoulos, 2020
"""
import os
import torch
import torch.nn as nn

import src


def train_model(dataset, model, writer=None):
    train_loader, valid_loader, test_loader = src.data.dataloader(dataset)
    data_shape = src.utils.get_data_shape(train_loader)

    model = nn.DataParallel(src.VAE(
        data_shape).to(src.utils.args.device), [0])
    # model = globals()[model](data_shape).to(args.device)
    model.module.initialize(train_loader)

    criterion = src.modules.loss.ELBOLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7)
    scheduler = src.modules.optim.LowerBoundedExponentialLR(
        optimizer, gamma=0.999999, lower_bound=0.0001)

    src.utils.n_parameters(model, writer)

    for epoch in range(1, src.utils.args.epochs):
        # Train and Validation epoch
        train_losses = src.modules.train(
            model, criterion, optimizer, scheduler, train_loader)
        valid_losses = src.modules.evaluate(
            model, criterion, valid_loader)
        # Visual Evaluation
        src.plotting.generate(model, src.utils.args.n_samples, epoch, writer)
        src.plotting.reconstruction(model, valid_loader,
                                    src.utils.args.n_samples, epoch, writer)
        # Saving Model and Loggin
        is_saved = src.utils.utils.save_model(
            model, optimizer, valid_losses['nelbo'], epoch)
        src.utils.utils.logging(epoch, train_losses,
                                valid_losses, is_saved, writer)


def resume_training(dataset, model):
    train_loader, valid_loader, test_loader = src.data.dataloader(dataset)
    data_shape = src.utils.get_data_shape(train_loader)
    writer = None

    pth = './src/models'
    pth = os.path.join(pth, 'pretrained',
                       src.utils.args.model, src.utils.args.dataset)
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    model = src.VAE(data_shape).to(src.utils.args.device)
    model.initialize(train_loader)

    criterion = src.modules.loss.ELBOLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7)

    checkpoint = torch.load(pth_train)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model successfully loaded!')
    except RuntimeError:
        print('? Fucked up loading model.')
        quit()
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Optimizer successfully loaded!')
    except RuntimeError:
        print('? Fucked up loading optimizer')
        quit()

    model = nn.DataParallel(model, [0]).to(src.utils.args.device)
    scheduler = src.modules.optim.LowerBoundedExponentialLR(
        optimizer, gamma=0.999999, lower_bound=0.0001)

    print('Somehow successfully resumed')
    print('Resuming training from {:d} epoch'.format(checkpoint['epoch']))

    for epoch in range(checkpoint['epoch'], src.utils.args.epochs):
        # Train and Validation epoch
        train_losses = src.modules.train(
            model, criterion, optimizer, scheduler, train_loader)
        valid_losses = src.modules.evaluate(
            model, criterion, valid_loader)
        # Visual Evaluation
        src.plotting.generate(model, src.utils.args.n_samples, epoch, writer)
        src.plotting.reconstruction(model, valid_loader,
                                    src.utils.args.n_samples, epoch, writer)
        # Saving Model and Loggin
        is_saved = src.utils.save_model(
            model, optimizer, valid_losses['nelbo'], epoch)
        src.utils.logging(epoch, train_losses, valid_losses, is_saved, writer)


def load_and_evaluate(dataset, model, writer=None):
    pth = './src/models/'

    # configure paths
    pth = os.path.join(pth, 'pretrained',
                       src.utils.args.model, src.utils.args.dataset)
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    # get data
    train_loader, valid_loader, test_loader = src.data.dataloaders.dataloader(
        dataset)
    data_shape = src.utils.get_data_shape(train_loader)

    # deifine model
    model = src.VAE(data_shape).to(src.utils.args.device)
    model.initialize(train_loader)

    # load trained weights for inference
    checkpoint = torch.load(pth_train)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model successfully loaded!')
    except RuntimeError:
        print('* Failed to load the model. Parameter mismatch.')
        quit()
    model = nn.DataParallel(model, [0]).to(src.utils.args.device)
    # model = model.to(args.device)
    model.eval()
    criterion = src.modules.loss.ELBOLoss()

    # Evaluation of the model
    # --- calculate elbo ---
    test_losses = src.modules.evaluate(model, criterion, test_loader)
    print('ELBO: {} bpd'.format(test_losses['bpd']))

    # --- image generation ---
    src.plotting.generate(model, n_samples=15*15)

    # --- image reconstruction ---
    src.plotting.reconstruction(model, test_loader, n_samples=15)

    # --- image interpolation ---
    src.plotting.interpolation(model, test_loader, n_samples=15)

    # --- calculate nll ---
    bpd = src.modules.loss.calculate_nll(
        model, test_loader, criterion,
        src.utils.args, iw_samples=src.utils.args.iw_test)
    print('NLL with {} weighted samples: {:4.2f}'.format(
        src.utils.args.iw_test, bpd))


# ----- main -----

def main():
    # Print configs
    src.utils.print_args(src.utils.args)

    # Control random seeds
    src.utils.fix_random_seed(seed=src.utils.args.seed)

    # Initialize TensorBoad writer (if enabled)
    writer = None

    # Train model
    # train_model(args.dataset, args.model, writer)
    resume_training(src.utils.args.dataset, src.utils.args.model)

    # Evaluate best (latest saved) model
    load_and_evaluate(src.utils.args.dataset, src.utils.args.model, writer)

    # End Experiment
    writer.close()
    print('\n'+24*'='+' Experiment Ended '+24*'=')


# ----- python main.py -----

if __name__ == "__main__":
    main()
