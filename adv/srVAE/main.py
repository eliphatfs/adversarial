""" PyTorch implimentation of VAE and Super-Resolution VAE.

    Reposetory Author:
        Ioannis Gatopoulos, 2020
"""
import os
import torch
import torch.nn as nn

from src import *


def train_model(dataset, model, writer=None):
    train_loader, valid_loader, test_loader = dataloader(dataset)
    data_shape = get_data_shape(train_loader)

    model = nn.DataParallel(globals()[model](data_shape).to(args.device), [0])
    # model = globals()[model](data_shape).to(args.device)
    model.module.initialize(train_loader)

    criterion = ELBOLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=2e-3, betas=(0.9, 0.999), eps=1e-7)
    scheduler = LowerBoundedExponentialLR(
        optimizer, gamma=0.999999, lower_bound=0.0001)

    n_parameters(model, writer)

    for epoch in range(1, args.epochs):
        # Train and Validation epoch
        train_losses = train(
            model, criterion, optimizer, scheduler, train_loader)
        valid_losses = evaluate(
            model, criterion, valid_loader)
        # Visual Evaluation
        generate(model, args.n_samples, epoch, writer)
        reconstruction(model, valid_loader,
                                args.n_samples, epoch, writer)
        # Saving Model and Loggin
        is_saved = save_model(
            model, optimizer, valid_losses['nelbo'], epoch)
        logging(epoch, train_losses, valid_losses, is_saved, writer)


def resume_training(dataset, model):
    train_loader, valid_loader, test_loader = dataloader(dataset)
    data_shape = get_data_shape(train_loader)
    writer = None

    pth = './src/models'
    pth = os.path.join(pth, 'pretrained', args.model, args.dataset)
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    model = globals()[model](data_shape).to(args.device)
    model.initialize(train_loader)

    criterion = ELBOLoss()
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

    model = nn.DataParallel(model, [0]).to(args.device)
    scheduler = LowerBoundedExponentialLR(
        optimizer, gamma=0.999999, lower_bound=0.0001)

    print('Somehow successfully resumed')
    print('Resuming training from {:d} epoch'.format(checkpoint['epoch']))

    for epoch in range(checkpoint['epoch'], args.epochs):
        # Train and Validation epoch
        train_losses = train(
            model, criterion, optimizer, scheduler, train_loader)
        valid_losses = evaluate(
            model, criterion, valid_loader)
        # Visual Evaluation
        generate(model, args.n_samples, epoch, writer)
        reconstruction(model, valid_loader,
                                args.n_samples, epoch, writer)
        # Saving Model and Loggin
        is_saved = save_model(
            model, optimizer, valid_losses['nelbo'], epoch)
        logging(epoch, train_losses, valid_losses, is_saved, writer)


def load_and_evaluate(dataset, model, writer=None):
    pth = './src/models/'

    # configure paths
    pth = os.path.join(pth, 'pretrained', args.model, args.dataset)
    pth_train = os.path.join(pth, 'trainable', 'model.pth')

    # get data
    train_loader, valid_loader, test_loader = dataloaders.dataloader(dataset)
    data_shape = get_data_shape(train_loader)

    # deifine model
    model = globals()[model](data_shape).to(args.device)
    model.initialize(train_loader)

    # load trained weights for inference
    checkpoint = torch.load(pth_train)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model successfully loaded!')
    except RuntimeError:
        print('* Failed to load the model. Parameter mismatch.')
        quit()
    model = nn.DataParallel(model, [0]).to(args.device)
    # model = model.to(args.device)
    model.eval()
    criterion = ELBOLoss()

    # Evaluation of the model
    # --- calculate elbo ---
    test_losses = evaluate(model, criterion, test_loader)
    print('ELBO: {} bpd'.format(test_losses['bpd']))

    # --- image generation ---
    generate(model, n_samples=15*15)

    # --- image reconstruction ---
    reconstruction(model, test_loader, n_samples=15)

    # --- image interpolation ---
    interpolation(model, test_loader, n_samples=15)

    # --- calculate nll ---
    bpd = calculate_nll(model, test_loader, criterion,
                        args, iw_samples=args.iw_test)
    print('NLL with {} weighted samples: {:4.2f}'.format(args.iw_test, bpd))


# ----- main -----

def main():
    # Print configs
    print_args(args)

    # Control random seeds
    fix_random_seed(seed=args.seed)

    # Initialize TensorBoad writer (if enabled)
    writer = None

    # Train model
    # train_model(args.dataset, args.model, writer)
    resume_training(args.dataset, args.model)

    # Evaluate best (latest saved) model
    load_and_evaluate(args.dataset, args.model, writer)

    # End Experiment
    writer.close()
    print('\n'+24*'='+' Experiment Ended '+24*'=')


# ----- python main.py -----

if __name__ == "__main__":
    main()
