import torch


def print_size(name, x):
    assert isinstance(name, str), "Name must be a string"
    assert isinstance(x, torch.Tensor), "Input must be a torch.Tensor"

    # Print the size of the tensor
    print(f"\033[33m{name}\033[0m" + f" --> {x.names}:  {x.size()}")

# a function that prints in yellow
def print_yellow(text):
    print(f"\033[33m{text}\033[0m")

# a function that prints in underlined
def print_underlined(text):
    print(f"\033[4m{text}\033[0m")