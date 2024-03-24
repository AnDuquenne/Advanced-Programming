import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import torch
import torch.nn as nn

import math

import yaml

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils.utils import print_size, print_underlined


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


class TimeSeriesTransformerForecaster(nn.Module):

    def __init__(self, feature_size=200, num_layers=8, expansion=4, dropout=0.1, nhead=10, debug=False):
        super(TimeSeriesTransformerForecaster, self).__init__()

        self.DROPOUT = dropout
        self.NHEAD = nhead
        self.NUM_LAYERS = num_layers
        self.DEBUG = debug
        self.EXPANSION = expansion

        self.pos_encoder = PositionalEncoding(d_model=feature_size, dropout=self.DROPOUT, debug=self.DEBUG)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=self.NHEAD, dropout=self.DROPOUT),
            num_layers=self.NUM_LAYERS
        )

        self.fc_decoder_1 = nn.Linear(feature_size, feature_size * self.EXPANSION)
        self.re1 = nn.ReLU()

        self.fc_decoder_2 = nn.Linear(feature_size * self.EXPANSION, feature_size)
        self.re2 = nn.ReLU()

        self.fc_decoder_3 = nn.Linear(feature_size, 1)

        # TODO Initialize the weights

    def forward(self, x):

        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)

        x = self.re1(self.fc_decoder_1(x))

        x = self.re2(self.fc_decoder_2(x))

        x = self.fc_decoder_3(x)

        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    d_model is the embedding size.
    """

    def __init__(self, d_model, max_len=1000, dropout=0.1, debug=False, debug_name=False):
        super(PositionalEncoding, self).__init__()

        assert d_model % 2 == 0, "Embedding size must be even"

        self.d_model = d_model
        self.max_len = max_len

        self.DEBUG = debug
        self.DEBUG_NAME = debug_name
        self.DROPOUT = dropout

        self.dropout = nn.Dropout(self.DROPOUT)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        assert x.size(0) <= self.max_len, "The input size is not correct"

        if self.DEBUG:

            print_underlined("Positional encoding")
            if self.DEBUG_NAME:
                x.names = ['batch_size', 'embedding_size', 'features']
                self.pe.names = ['batch_size', 'features', 'one_dim']
            print_size("x", x)
            print_size("pe", self.pe)
            print_size("pe_resized", self.pe[:x.size(0), :])
            print_size("x + pe_resized", x + self.pe[:x.size(0), :])

        return self.dropout(x + self.pe[:x.size(0), :])


if __name__ == '__main__':
    # Load the configuration file
    with open('../io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the torch data
    X = torch.load('../' + config['Strategy']['Transformers']['data_path_X'])
    y = torch.load('../' + config['Strategy']['Transformers']['data_path_y'])

    save_path_weights = '../' + config['Strategy']['Transformers']['weights_path']
    save_path_loss = '../' + config['Strategy']['Transformers']['loss_path']

    DEBUG = config['Strategy']['Transformers']['debug']
    DEBUG_NAME = config['Strategy']['Transformers']['debug_name']

    LOAD_WEIGHTS = config['Strategy']['Transformers']['load_weights']
    BATCH_SIZE = config['Strategy']['Transformers']['batch_size']
    LEARNING_RATE = config['Strategy']['Transformers']['learning_rate']
    NB_EPOCHS = config['Strategy']['Transformers']['nb_epochs']

    EMBEDDING_SIZE = config['Strategy']['Transformers']['embedding_size']
    KERNEL_SIZE = config['Strategy']['Transformers']['kernel_size']
    NB_HEADS = config['Strategy']['Transformers']['nb_heads']
    NB_LAYERS = config['Strategy']['Transformers']['nb_encoder_layers']
    FORWARD_EXPANSION = config['Strategy']['Transformers']['forward_expansion']
    FORWARD_EXPANSION_DECODER = config['Strategy']['Transformers']['forward_expansion_decoder']
    MAX_LEN = config['Strategy']['Transformers']['max_len']

    DROPOUT = config['Strategy']['Transformers']['dropout']
    NORMALIZE = config['Strategy']['Transformers']['normalize']

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = f"Transformers_ES_{EMBEDDING_SIZE}_NH_{NB_HEADS}_FE_{FORWARD_EXPANSION}_NBL_{NB_LAYERS}"

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    if DEBUG:
        # Name the dimensions
        print_underlined("\n Data recovered from the folder")

        if DEBUG_NAME:
            X_train.names = ['n_observations', 'features', 'one_dim']
            y_train.names = ['n_observations', 'features']
            X_test.names = ['n_observations', 'features', 'one_dim']
            y_test.names = ['n_observations', 'features']

        print_size("X_train", X_train)
        print_size("y_train", y_train)
        print_size("X_test", X_test)
        print_size("y_test", y_test)

        if DEBUG_NAME:
            X_train = X_train.rename(None)
            y_train = y_train.rename(None)
            X_test = X_test.rename(None)
            y_test = y_test.rename(None)

    model = TimeSeriesTransformerForecaster(
        feature_size=EMBEDDING_SIZE,
        num_layers=NB_LAYERS,
        dropout=DROPOUT,
        nhead=NB_HEADS,
        expansion=FORWARD_EXPANSION,
        debug=DEBUG
    ).to(DEVICE)

    if LOAD_WEIGHTS:
        try:
            model.load_state_dict(torch.load(save_path_weights + MODEL_NAME + ".pt"))
        except:
            print("No weights found")

    # The data loader
    train_loader = DataLoader(TimeSeriesDataframe(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataframe(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize the loss history
    epoch_train_loss = np.zeros((NB_EPOCHS, 1))
    epoch_test_loss = np.zeros((NB_EPOCHS, 1))

    for i in tqdm(range(0, NB_EPOCHS)):

        # Initialize the loss for each epoch
        tmp_train_loss = np.zeros((len(train_loader), 1))
        tmp_test_loss = np.zeros((len(test_loader), 1))

        for idx, (signal, target) in enumerate(train_loader):

            # Set the model to train mode
            model.train()

            # We want [window, batch_size, features]
            signal = signal.permute(2, 0, 1)
            target = target.permute(2, 0, 1)

            signal = signal.float()
            target = target.float()

            # Move the data to the device
            signal = signal.to(DEVICE)
            target = target.to(DEVICE)

            # Reset grad
            optimizer.zero_grad()
            # Make predictions
            preds = model(signal.to(DEVICE))

            if DEBUG:
                print_underlined("Data in the training loop (Coming from the dataloader)")
                if DEBUG_NAME:
                    signal.names = ['batch_size', 'features', 'one_dim']
                    preds.names = ['batch_size', 'features', 'output_size']
                    target.names = ['batch_size', 'output_size']

                print_size("signal", signal)
                print_size("preds", preds)
                print_size("target", target)

                if DEBUG_NAME:
                    signal = signal.rename(None)
                    preds = preds.rename(None)
                    target = target.rename(None)

            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

            tmp_test_loss_ = np.zeros((len(test_loader), 1))

        model.eval()
        with torch.no_grad():
            for idx, (signal, target) in enumerate(test_loader):
                # We want [window, batch_size, features]
                signal = signal.permute(2, 0, 1)
                target = target.permute(2, 0, 1)

                signal = signal.float()
                target = target.float()

                signal = signal.to(DEVICE)
                target = target.to(DEVICE)

                preds = model(signal.to(DEVICE))

                if idx == i:
                    print(preds.cpu().detach().size())
                    print(signal.cpu().detach().size())
                    print(target.cpu().detach().size())
                    plt.plot(preds[:, idx, :].cpu().detach().numpy(), color='red')
                    plt.plot(signal[:, idx, 0].cpu().detach().numpy(), color='green')
                    plt.plot(target[:, idx, :].cpu().detach().numpy(), color='blue')
                    plt.show()

                loss = criterion(preds, target)

                tmp_test_loss[idx] = np.mean(loss.cpu().detach().item())

        epoch_train_loss[i] = np.mean(tmp_train_loss)
        epoch_test_loss[i] = np.mean(tmp_test_loss)

    # Save the model
    torch.save(model.state_dict(), save_path_weights + MODEL_NAME + ".pt")

    # save the loss
    np.save(save_path_loss + MODEL_NAME + "_train_loss.npy", epoch_train_loss)
    np.save(save_path_loss + MODEL_NAME + "_test_loss.npy", epoch_test_loss)

    # Plot the loss
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_test_loss, label='Test Loss')
    # Add grid
    plt.grid()
    plt.legend()
    # save the plot
    plt.savefig(save_path_loss + MODEL_NAME + ".png")
    plt.show()
