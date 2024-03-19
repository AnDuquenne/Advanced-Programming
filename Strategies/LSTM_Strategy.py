import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import yaml

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

class TimeSeriesDataframe(Dataset):
    """
    A class to create a time series dataframe.
    """

    def __init__(self, X, y):
        self.X = None
        self.y = None
    def __len__(self):
        return len(self.torch_df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM:
    def __init__(self, input_size, hidden_size, num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


if __name__ == '__main__':
    # Load the configuration file
    with open('../io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the torch data
    X = torch.load('../' + config['Strategy']['LSTM']['data_path_X'])
    y = torch.load('../' + config['Strategy']['LSTM']['data_path_y'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    batch_size = config['Strategy']['LSTM']['batch_size']

    model = LSTM(input_size=1, hidden_size=100, num_layers=1)

    train_loader = DataLoader(TimeSeriesDataframe(), batch_size=batch_size, shuffle=True)
