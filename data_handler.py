import pandas as pd
import numpy as np

import yaml
import os

# Import deep copy
from copy import deepcopy as dc

from tqdm import tqdm

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from utils.technical_analysis import StochasticRSI, MACD, DPO, RSI, CC

from torch.utils.data import Dataset, DataLoader


class DataCleaner:
    """
        Clean the data and add the technical indicators.
    """

    def __init__(self, to_load, params_path):
        # Import yaml file
        with open(params_path, 'r') as file:
            config = yaml.safe_load(file)

        self.to_load = to_load
        self.params_path = params_path
        self.load_path = config['data_path'][f'{self.to_load}']['1min']['raw']
        self.save_path = config['data_path'][f'{self.to_load}']['1min']['cleaned']
        self.save_neural_path_LSTM = config['data_path'][f'{self.to_load}']['1min']['neural_LSTM']
        self.save_neural_path_transformers = config['data_path'][f'{self.to_load}']['1min']['neural_transformers']
        self.files = self.get_files()

    def get_files(self):
        """
        Get the files in the raw data folder.
        """
        # list of the files
        files = os.listdir(self.load_path)
        return files

    def clean_data(self):
        """
        Clean the data and add the technical indicators.
        """
        for file in tqdm(self.files):
            # Load the data
            data = pd.read_csv(self.load_path + file)

            # Ignore the warning of invalid value encountered in true_divide
            np.seterr(divide='ignore', invalid='ignore')

            # Create the MACD
            macd = MACD(data['close'])
            data['MACD'] = macd.get_macd
            data['Signal Line'] = macd.get_signal_line
            data['Histogram'] = macd.get_histogram

            data['RSI'] = RSI(data['close'], 14).get_rsi
            data['Stochastic RSI'] = StochasticRSI(data['close'], 14).get_stochastic_rsi
            data['DPO'] = DPO(data['close'], 21).get_dpo
            data['CC'] = CC(data['close'], 14).get_cc

            # Remove the first 3240 rows, the MACD tends to the good value after 3240 rows,
            # before, it is biased due to lack of historical values
            data = data.iloc[3240:]

            # Save the data
            data.to_csv(self.save_path + file, index=False)

    def prepare_dataframe_LSTM(self, window, look_forward=1):

        assert look_forward < window, "The look_forward parameter must be less than the window parameter."

        for file in tqdm(self.files):
            df = pd.read_csv(self.save_path + file)
            df = df[['date', 'close', 'MACD', 'Signal Line', 'Histogram', 'RSI', 'Stochastic RSI', 'DPO', 'CC']]
            df.set_index('date', inplace=True)

            df_numpy = dc(df).to_numpy()

            # # Minmaxscaler to scale the data expect the "Close" column
            # scaler = MinMaxScaler(feature_range=(-1, 1))
            # df_numpy[1:, :] = scaler.fit_transform(df_numpy[1:, :])

            X_torch = torch.zeros(df_numpy.shape[0] - window, df_numpy.shape[1] * window)
            for i in range(df_numpy.shape[1]):
                for j in range(window):
                    X_torch[:, (i * window) + j] = torch.tensor(df_numpy[j: -window + j, i])

            y_torch = dc(X_torch[:, window-look_forward:window])


            # Remove the last look_forward columns for the X_torch and the technical indicators
            df_X = pd.DataFrame(X_torch.numpy())
            df_y = pd.DataFrame(y_torch.numpy())

            cols_to_drop = []
            for i in range(df_numpy.shape[1]):
                for j in range(look_forward):
                    cols_to_drop.append((i * window) + (window - j) - 1)

            print(cols_to_drop)

            # Create a boolean mask where True indicates the columns we want to keep
            mask = torch.ones(X_torch.shape[1], dtype=torch.bool)  # Initially set all to True
            mask[cols_to_drop] = False  # Set the columns we want to drop to False

            # Drop the columns in torch_X
            # Use the mask to select columns
            X_torch = X_torch[:, mask]


            # Remove the last "look_forward" rows for the x_torch and y_torch
            # (due to shift in dataset creation, the last "look_forward" rows are not valid)
            X_torch = X_torch[:-look_forward]
            y_torch = y_torch[:-look_forward]

            # Save the data as torch tensors
            torch.save(X_torch, self.save_neural_path + file.replace('.csv', '_X.pt'))
            torch.save(y_torch, self.save_neural_path + file.replace('.csv', '_y.pt'))

    def prepare_dataframe_transformers(self, window, look_forward=1, log_returns=False):
        assert look_forward < window, "The look_forward parameter must be less than the window parameter."

        for file in tqdm(self.files):
            df = pd.read_csv(self.save_path + file)
            df = df[['date', 'close', 'MACD', 'Signal Line', 'Histogram', 'RSI', 'Stochastic RSI', 'DPO', 'CC']]
            df.set_index('date', inplace=True)

            df_numpy = dc(df).to_numpy()

            if log_returns:
                # Transform the closing prices to log returns
                df_numpy[1:, 0] = np.diff(np.log(df_numpy[:, 0]))

            # Minmaxscaler to scale the data expect the "Close" column
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df_numpy[1:, :] = scaler.fit_transform(df_numpy[1:, :])

            # Remove first row
            df_numpy = df_numpy[1:, :]

            # Transpose the numpy array
            df_numpy = df_numpy.T

            # Create the numpy data array (n_samples, n_features, window)
            # It contains the original data and the technical indicators
            numpy_windowed_data = np.zeros((df_numpy.shape[1] - window, df_numpy.shape[0], window))

            for i in range(df_numpy.shape[1] - window):
                numpy_windowed_data[i, :, :] = df_numpy[:, i:i+window]

            # Create the torch tensors, for y we keep only the feature "Close"
            X_torch = torch.tensor(numpy_windowed_data[:-look_forward, :, :])
            y_torch = torch.tensor(numpy_windowed_data[look_forward:, 0, :])

            # For DEBUG view
            # test_X = pd.DataFrame(dc(X_torch[35, :, :]).numpy())
            # test_y = pd.DataFrame(dc(y_torch[35, :]).numpy())

            # Save the data as torch tensors
            torch.save(X_torch, self.save_neural_path_transformers + file.replace('.csv', '_X.pt'))
            torch.save(y_torch, self.save_neural_path_transformers + file.replace('.csv', '_y.pt'))


if __name__ == '__main__':
    # Load the configuration file
    with open('io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Clean the data
    cleaner = DataCleaner('BTC', 'io/config.yaml')
    # cleaner.clean_data()
    cleaner.prepare_dataframe_transformers(10, 3, log_returns=True)
