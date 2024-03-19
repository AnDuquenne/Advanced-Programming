import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import torch
import torch.nn as nn

import yaml

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


class TimeSeriesDataframe(Dataset):
    """
    A class to create a time series dataframe.
    """

    def __init__(self, X, y):
        
        super().__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, fc_out, dropout):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.fc_out = fc_out
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_size, self.fc_out)

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

    save_path = '../' + config['Strategy']['LSTM']['weights_path']

    BATCH_SIZE = config['Strategy']['LSTM']['batch_size']
    LEARNING_RATE = config['Strategy']['LSTM']['learning_rate']
    NB_EPOCHS = config['Strategy']['LSTM']['nb_epochs']
    INPUT_SIZE = config['Strategy']['LSTM']['input_size']
    LOAD_WEIGHTS = config['Strategy']['LSTM']['load_weights']
    HIDDEN_SIZE = config['Strategy']['LSTM']['hidden_size']
    NUM_LAYERS = config['Strategy']['LSTM']['num_layers']
    DROPOUT = config['Strategy']['LSTM']['dropout']

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = f"LSTM_IS_{INPUT_SIZE}_HS_{HIDDEN_SIZE}_NL_{NUM_LAYERS}_DO_{DROPOUT}"

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.unsqueeze(2)
    X_test = X_test.unsqueeze(2)

    model = LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_out=y_train.shape[1],
                 dropout=DROPOUT, device=DEVICE).to(DEVICE)

    if LOAD_WEIGHTS:
        try:
            model.load_state_dict(torch.load(save_path + MODEL_NAME + ".pt"))
        except:
            print("No weights found")

    train_loader = DataLoader(TimeSeriesDataframe(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataframe(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_train_loss = np.zeros((NB_EPOCHS, 1))
    epoch_test_loss = np.zeros((NB_EPOCHS, 1))

    for i in tqdm(range(0, NB_EPOCHS)):

        tmp_train_loss = np.zeros((len(train_loader), 1))
        tmp_test_loss = np.zeros((len(test_loader), 1))

        for idx, (signal, target) in enumerate(train_loader):

            model.train()

            signal = signal.to(DEVICE)
            target = target.to(DEVICE)

            # Reset grad
            optimizer.zero_grad()
            # Make predictions
            preds = model(signal.to(DEVICE))

            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

            tmp_test_loss_ = np.zeros((len(test_loader), 1))

        model.eval()
        with torch.no_grad():
            for idx, (signal, target) in enumerate(test_loader):
                signal = signal.to(DEVICE)
                target = target.to(DEVICE)

                preds = model(signal.to(DEVICE))

                loss = criterion(preds, target)

                tmp_test_loss[idx] = np.mean(loss.cpu().detach().item())

        epoch_train_loss[i] = np.mean(tmp_train_loss)
        epoch_test_loss[i] = np.mean(tmp_test_loss)

    # Save the model
    torch.save(model.state_dict(), save_path + MODEL_NAME + ".pt")

    # Plot the loss
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_test_loss, label='Test Loss')
    plt.legend()
    plt.show()