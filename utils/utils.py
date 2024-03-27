import torch
import datetime

import os
import shutil

def print_size(name, x):
    assert isinstance(name, str), "Name must be a string"
    assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"

    # Print the size of the tensor
    print(f"\033[33m{name}\033[0m" + f" --> {x.names}:  {x.size()}")

# a function that prints in yellow
def print_yellow(text):
    print(f"\033[33m{text}\033[0m")

def print_red(text):
    print(f"\033[31m{text}\033[0m")

def print_green(text):
    print(f"\033[32m{text}\033[0m")

# a function that prints in underlined
def print_underlined(text):
    print(f"\033[4m{text}\033[0m")

def train_test_split_custom(X, y, train_size=0.8):
    assert X.size(0) == y.size(0), "The number of samples in X and y must be the same"
    assert 0 < train_size < 1, "The train_size must be between 0 and 1"

    # Calculate the number of samples for the training set
    train_samples = int(X.size(1) * train_size)

    # Split the data
    X_train, X_test = X[:, :train_samples, :], X[:, train_samples:, :]
    y_train, y_test = y[:, :train_samples, :], y[:, train_samples:, :]

    return X_train, X_test, y_train, y_test


def mkdir_save_model(save_path):
    # Create a folder to save the weights and the loss
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/weights")
        os.makedirs(save_path + "/loss")

    # Create a copy of the config.yaml file in the same folder
    shutil.copy("../io/config.yaml ", save_path + "/")


def log_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    return total_norm
