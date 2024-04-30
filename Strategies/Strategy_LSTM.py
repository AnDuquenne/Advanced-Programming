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
from utils.market import *

# Load environment variables
import os
import sys
from dotenv import load_dotenv

env = os.getenv("environment")


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


class NetworkLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, fc_out, dropout):
        super(NetworkLSTM, self).__init__()

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

    def predict(self, ts, forward):
        """
        Make prediction for "forward" days
        :param ts: the time series dim 1
        :param forward: number of days
        :return: the prediction for the next "forward" days
        """
        pred_ts = torch.zeros(ts.size(0) + forward)
        pred_ts[:ts.size(0)] = ts
        # print(pred_ts)
        pred_ts = pred_ts.unsqueeze(0).unsqueeze(1)
        pred_ts = pred_ts.float().to(self.device)
        with torch.no_grad():
            for i in range(forward):
                pred = self.forward(pred_ts[:, :, i:i + ts.size(0)])
                pred_ts[:, :, ts.size(0) + i] = pred

        # print(ts)
        # print(pred_ts)
        # Squeeze the tensor pred_ts
        pred_ts = pred_ts.squeeze()
        # print(pred_ts)

        return pred_ts[-forward:]

    def load_weights(self, path):
        if env == 'local':
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=self.device))

class StrategyLSTM:
    def __init__(self, run_name, network_params, buy_percentage=0.01, exposure=2, debug=True):
        """
        Initialize the strategy

        :param run_name: Name of the run
        :param buy_percentage: The percentage range to create the orders
        ex. if buy_percentage = 0.01, the orders will be created at 1% intervals (200, 198, ...)
        :param exposure: Wallet exposure to the orders
        ex. if exposure = 0.5, 50% of the wallet is open to create the order book
        """
        self.run_name = run_name
        self.buy_percentage = buy_percentage
        self.exposure = exposure

        INPUT_SIZE = network_params['input_size']
        HIDDEN_SIZE = network_params['hidden_size']
        NUM_LAYERS = network_params['num_layers']
        DROPOUT = network_params['dropout']
        DEVICE = network_params['device']
        weights_path = network_params['weights_path']

        self.forecaster = NetworkLSTM(input_size=INPUT_SIZE,
                                      hidden_size=HIDDEN_SIZE,
                                      num_layers=NUM_LAYERS,
                                      fc_out=1,
                                      dropout=DROPOUT,
                                      device=DEVICE).to(DEVICE)

        self.forecaster.load_weights(weights_path)

    def check_conditions(self, orders, positions, data, time, wallet, forward=5):
        """
        Central part of the strategy class. It takes as input the list of positions and orders and the data.
        The data is composed of the prices of the index as well as what the strategy needs to make a decision.
        In this case, the strategy is very simple and only needs the closing prices.

        :param orders: A list of the orders
        :param positions: A list of the positions
        :param data: The index prices (10 historical values)
        :param time: The time of the index price
        :param wallet: The amount disposable to trade
        :param forward: The number of days to predict
        :return: The update lists of orders and positions and wallet
        """

        assert len(data) == 10, "The data should contain 10 historical values"
        assert isinstance(data, pd.Series), "The data should be a pandas Series"
        assert isinstance(wallet, float), "The wallet should be a float"

        order_list = orders
        position_list = positions
        wallet = wallet

        # Transform the data to a 1D tensor
        data = torch.tensor(data.values).float()

        # Get the prediction
        pred = self.forecaster.predict(data, forward)

        # Using predictions to take decisions
        curr_price = data[-1].item()

        # Amount to buy
        am_ = wallet * self.exposure / 100 / curr_price

        # If enough money to buy
        if wallet > am_ * curr_price:
            # If the next price is lower than the current price,
            # and one of the last 5 predictions is higher than the current price then buy
            min_pred = min(pred)
            max_pred = max(pred)
            if curr_price * (1 + self.buy_percentage) < min_pred:
                position_list.append(Position(
                    opening_price=curr_price,
                    opening_time=time,
                    amount=am_,
                    direction='long',
                    closing_price=curr_price * (1 + self.buy_percentage))
                    # closing_price=max_pred)
                )

                # Update the wallet value
                wallet -= am_ * curr_price

        else:
            print("Not enough money to buy")

        # Check positions
        dollars_value = 0
        for pos in position_list:
            if pos.status == 'open':
                if pos.closing_price <= curr_price:
                    pos.close(curr_price, time)
                    dollars_value += pos.dollars_value(curr_price)

        # Update the wallet
        wallet += dollars_value

        return order_list, position_list, wallet

    def __str__(self):
        return "StrategyLSTM" + self.run_name

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

    # cleaner = DataCleaner('ETH', '../io/config.yaml')
    # cleaner.prepare_dataframe_FC(
    #     window=WINDOW,
    #     look_forward=LOOK_FORWARD,
    #     log_close=LOG_CLOSE,
    #     close_returns=CLOSE_RETURNS,
    #     only_close=ONLY_CLOSE,
    #     min_max_scale=MIN_MAX_SCALING
    # )

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

    model = NetworkLSTM(input_size=X.size(2), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, fc_out=y_train.size(1),
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
