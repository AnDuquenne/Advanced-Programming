import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as datetime

from datetime import datetime

import torch
import torch.nn as nn

import yaml

import wandb

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from utils.data_handler import DataCleaner
from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.trainer_LSTM import Trainer


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


class Strategy_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, fc_out, dropout):
        super(Strategy_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.fc_out = fc_out
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=self.dropout, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.fc_out)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


if __name__ == '__main__':

    # ------------------------------------------------------------------------ #
    #                              Load the params                             #
    # ------------------------------------------------------------------------ #
    with open('../io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    SAVE_PATH_LOSS = '../' + config['Strategy']['LSTM']['loss_path']
    SAVE_PATH_WEIGHTS = '../' + config['Strategy']['LSTM']['weights_path']
    SAVE_PATH = '../' + config['Strategy']['LSTM']['save_path']

    BATCH_SIZE = config['Strategy']['LSTM']['batch_size']
    LEARNING_RATE = config['Strategy']['LSTM']['learning_rate']
    NB_EPOCHS = config['Strategy']['LSTM']['nb_epochs']
    INPUT_SIZE = config['Strategy']['LSTM']['input_size']
    HIDDEN_SIZE = config['Strategy']['LSTM']['hidden_size']
    NUM_LAYERS = config['Strategy']['LSTM']['num_layers']
    DROPOUT = config['Strategy']['LSTM']['dropout']

    WINDOW = config['Strategy']['LSTM']['dh_params']['window']
    LOOK_FORWARD = config['Strategy']['LSTM']['dh_params']['look_forward']
    LOG_CLOSE = config['Strategy']['LSTM']['dh_params']['log_close']
    CLOSE_RETURNS = config['Strategy']['LSTM']['dh_params']['close_returns']
    ONLY_CLOSE = config['Strategy']['LSTM']['dh_params']['only_close']
    MIN_MAX_SCALING = config['Strategy']['LSTM']['dh_params']['min_max_scaling']

    DEBUG = config['Strategy']['LSTM']['debug']
    WANDB = config['Strategy']['LSTM']['wandb']
    LOAD_WEIGHTS = config['Strategy']['LSTM']['load_weights']

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = f"LSTM_IS_{INPUT_SIZE}_HS_{HIDDEN_SIZE}_NL_{NUM_LAYERS}_DO_{DROPOUT}"

    # ------------------------------------------------------------------------ #
    #                              Create the data                             #
    # ------------------------------------------------------------------------ #

    cleaner = DataCleaner('BTC', '../io/config.yaml')
    cleaner.prepare_dataframe_LSTM(
        window=WINDOW,
        look_forward=LOOK_FORWARD,
        log_close=LOG_CLOSE,
        close_returns=CLOSE_RETURNS,
        only_close=ONLY_CLOSE,
        min_max_scale=MIN_MAX_SCALING
    )

    cleaner = DataCleaner('ETH', '../io/config.yaml')
    cleaner.prepare_dataframe_LSTM(
        window=WINDOW,
        look_forward=LOOK_FORWARD,
        log_close=LOG_CLOSE,
        close_returns=CLOSE_RETURNS,
        only_close=ONLY_CLOSE,
        min_max_scale=MIN_MAX_SCALING
    )

    # ------------------------------------------------------------------------ #
    #                        Load and process the data                         #
    # ------------------------------------------------------------------------ #

    # Load the torch data
    X = torch.load('../' + config['Strategy']['LSTM']['data_path_X'])
    y = torch.load('../' + config['Strategy']['LSTM']['data_path_y'])

    # Split the data (do not shuffle, will be shuffled in dataloader)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ------------------------------------------------------------------------ #
    #                             Define the model                             #
    # ------------------------------------------------------------------------ #

    model = Strategy_LSTM(input_size=X.size(2), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_out=y_train.size(1),
               dropout=DROPOUT, device=DEVICE).to(DEVICE)

    # ------------------------------------------------------------------------ #
    #                               Load weights                               #
    # ------------------------------------------------------------------------ #

    # Procedure to load weights:
    # 1. Check in the IO models if models are saved with the same parameters in the config file
    # 2. If yes, load the weights for the most recent model
    if LOAD_WEIGHTS:
        print_underlined("Loading weights")
        # Create a list of all folders in the 'SAVE_PATH' folder
        folders = os.listdir(SAVE_PATH)
        for folder in reversed(folders):
            # try to open the config file in the folder
            try:
                with open(SAVE_PATH + folder + "/config.yaml", 'r') as file:
                    config_folder = yaml.safe_load(file)
                    if config_folder['Strategy']['LSTM']['hidden_size'] == HIDDEN_SIZE and \
                            config_folder['Strategy']['LSTM']['num_layers'] == NUM_LAYERS and \
                            config_folder['Strategy']['LSTM']['dh_params']['window'] == WINDOW and \
                            config_folder['Strategy']['LSTM']['dh_params'][
                                'look_forward'] == LOOK_FORWARD and \
                            config_folder['Strategy']['LSTM']['dh_params']['log_close'] == LOG_CLOSE and \
                            config_folder['Strategy']['LSTM']['dh_params'][
                                'close_returns'] == CLOSE_RETURNS and \
                            config_folder['Strategy']['LSTM']['dh_params']['only_close'] == ONLY_CLOSE:
                        # Load the weights
                        model.load_state_dict(torch.load(SAVE_PATH + folder + "/weights/weights.pt"))
                        print_green(f"Loaded weights from {folder}")
                        break
            except:
                print_red("Failed to load the weights")
    # ------------------------------------------------------------------------ #
    #                              Train the model                             #
    # ------------------------------------------------------------------------ #

    train_loader = DataLoader(TimeSeriesDataframe(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataframe(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.975)

    trainer = Trainer(train_loader, test_loader, model, optimizer, criterion, scheduler, DEVICE, NB_EPOCHS,
                      SAVE_PATH_LOSS, SAVE_PATH_WEIGHTS, MODEL_NAME, DEBUG, SAVE_PATH, WANDB)

    trainer.train()
    # trainer.evaluate(X_test[500:700, :, :], y_test[500:700, :, :])
