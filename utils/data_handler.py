import os, sys
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

import pandas as pd
import numpy as np

import yaml

# Import deep copy
from copy import deepcopy as dc

from tqdm import tqdm

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from technical_analysis import *

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
        self.save_neural_path_fc = config['data_path'][f'{self.to_load}']['1min']['neural_fc']
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

    def prepare_dataframe_FC(self, window, look_forward=1, log_close=False, close_returns=False, only_close=False,
                             min_max_scale=False):

        for file in tqdm(self.files):
            df = pd.read_csv(self.save_path + file)
            df = df[['date', 'close', 'MACD', 'Signal Line', 'Histogram', 'RSI', 'Stochastic RSI', 'DPO', 'CC']]
            df.set_index('date', inplace=True)

            df_numpy = dc(df).to_numpy()

            if min_max_scale:
                # Scale the data over columns
                scaler = MinMaxScaler()
                df_numpy = scaler.fit_transform(df_numpy)

            if log_close:
                # Transform the closing prices to log prices
                df_numpy[:, 0] = np.log(df_numpy[:, 0])

            if close_returns:
                # Calculate the returns as being the ratio between the log prices
                df_numpy[1:, 0] = df_numpy[1:, 0] / df_numpy[:-1, 0]
                # Remove the first row
                df_numpy = df_numpy[1:, :]
                # Remove 1, to have a percentage change
                df_numpy[:, 0] -= 1
                df_numpy[:, 0] *= 100

            if only_close:
                df_numpy = df_numpy[:, 0].reshape(-1, 1)


            # Transpose the numpy array
            df_numpy = df_numpy.T

            ts_length = df_numpy.shape[1]
            nb_features = df_numpy.shape[0]

            # Create the numpy data array (nb of possible windows, n_features, window + look_forward)
            # It contains the original data and the technical indicators
            numpy_windowed_data = np.zeros((ts_length - (window + look_forward),
                                            nb_features,
                                            window + look_forward))

            for i in range(ts_length - (window + look_forward)):
                numpy_windowed_data[i, :, :] = df_numpy[:, i:i + window + look_forward]

            # Numpy windowed data is [nb of possible windows, nb of features, window size]

            # Create the torch tensors, for y we keep only the feature "Close"
            # S: sequence length, N: batch size, E: number of features (or embedding size)
            X_torch = torch.tensor(numpy_windowed_data[:, :, :(window - look_forward + 1)])
            y_torch = torch.tensor(numpy_windowed_data[:, 0, (window - look_forward + 1):]).unsqueeze(1)

            # Save the data as torch tensors
            torch.save(X_torch, self.save_neural_path_fc + file.replace('.csv', '_X.pt'))
            torch.save(y_torch, self.save_neural_path_fc + file.replace('.csv', '_y.pt'))

    def prepare_dataframe_LSTM(self, window, look_forward=1):

        assert look_forward < window, "The look_forward parameter must be less than the window parameter."

        for file in tqdm(self.files):
            df = pd.read_csv(self.save_path + file)
            df = df[['date', 'close', 'MACD', 'Signal Line', 'Histogram', 'RSI', 'Stochastic RSI', 'DPO', 'CC']]
            df.set_index('date', inplace=True)

            df_numpy = dc(df).to_numpy()

            X_torch = torch.zeros(df_numpy.shape[0] - window, df_numpy.shape[1] * window)
            for i in range(df_numpy.shape[1]):
                for j in range(window):
                    X_torch[:, (i * window) + j] = torch.tensor(df_numpy[j: -window + j, i])

            y_torch = dc(X_torch[:, window - look_forward:window])

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
            torch.save(X_torch, self.save_neural_path_LSTM + file.replace('.csv', '_X.pt'))
            torch.save(y_torch, self.save_neural_path_LSTM + file.replace('.csv', '_y.pt'))

    def prepare_dataframe_transformers(self, window, look_forward=1, decoder_horizon=1,
                                       log_close=False, close_returns=False, only_close=False, min_max_scale=False):
        assert look_forward < window, "The look_forward parameter must be less than the window parameter."

        for file in tqdm(self.files):
            df = pd.read_csv(self.save_path + file)
            df = df[['date', 'close', 'MACD', 'Signal Line', 'Histogram', 'RSI', 'Stochastic RSI', 'DPO', 'CC']]
            df.set_index('date', inplace=True)

            # df['MACD'] /= df['MACD'].max();
            # df['MACD'] += 1
            # df['Signal Line'] /= df['Signal Line'].max();
            # df['Signal Line'] += 1
            # df['Histogram'] /= df['Histogram'].max();
            # df['Histogram'] += 1
            # df['RSI'] /= df['RSI'].max();
            # df['RSI'] += 1
            # df['Stochastic RSI'] /= df['Stochastic RSI'].max();
            # df['Stochastic RSI'] += 1
            # df['DPO'] /= df['DPO'].max();
            # df['DPO'] += 1
            # df['CC'] /= df['CC'].max();
            # df['CC'] += 1

            df_numpy = dc(df).to_numpy()

            if min_max_scale:
                # Scale the data over columns
                scaler = MinMaxScaler()
                df_numpy = scaler.fit_transform(df_numpy)

            if log_close:
                # Transform the closing prices to log prices
                df_numpy[:, 0] = np.log(df_numpy[:, 0])

            if close_returns:
                # Calculate the returns as being the ratio between the log prices
                df_numpy[1:, 0] = df_numpy[1:, 0] / df_numpy[:-1, 0]
                # Remove the first row
                df_numpy = df_numpy[1:, :]
                # Remove 1, to have a percentage change
                df_numpy[:, 0] -= 1
                df_numpy[:, 0] *= 100

            if only_close:
                df_numpy = df_numpy[:, 0].reshape(-1, 1)

            # Transpose the numpy array
            df_numpy = df_numpy.T

            # Create the numpy data array (nb of possible windows, n_features, window + look_forward)
            # It contains the original data and the technical indicators
            numpy_windowed_data = np.zeros((df_numpy.shape[1] - (window + look_forward),
                                            df_numpy.shape[0],
                                            window + look_forward))

            for i in range(df_numpy.shape[1] - (window + look_forward)):
                numpy_windowed_data[i, :, :] = df_numpy[:, i:i + window + look_forward]

            # Numpy windowed data is [nb of possible windows, nb of features, window size]

            # Create the torch tensors, for y we keep only the feature "Close"
            # For the transformer layers we need (S, N, E)
            # S: sequence length, N: batch size, E: number of features (or embedding size)
            X_torch_encoder = torch.tensor(numpy_windowed_data[:, :, :(window - 1)])
            X_torch_decoder = torch.tensor(numpy_windowed_data[:, 0, (window - 2):-look_forward]).unsqueeze(1)
            y_torch = torch.tensor(numpy_windowed_data[:, 0, (window - 1):]).unsqueeze(1)

            # Get [N_windows, seq_len, n_features]
            X_torch_encoder = X_torch_encoder.transpose(1, 2)
            X_torch_decoder = X_torch_decoder.transpose(1, 2)
            y_torch = y_torch.transpose(1, 2)

            # Save the data as torch tensors
            torch.save(X_torch_encoder,
                       self.save_neural_path_transformers + file.replace('.csv', '_X_encoder.pt'))
            torch.save(X_torch_decoder,
                       self.save_neural_path_transformers + file.replace('.csv', '_X_decoder.pt'))
            torch.save(y_torch, self.save_neural_path_transformers + file.replace('.csv', '_y.pt'))


if __name__ == '__main__':
    # Load the configuration file
    with open('../io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Clean the data
    cleaner = DataCleaner('ETH', '../io/config.yaml')
    # cleaner.clean_data()
    cleaner.prepare_dataframe_FC(10, 1)
