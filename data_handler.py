import pandas as pd
import numpy as np

import yaml
import os

from tqdm import tqdm

from utils.technical_analysis import StochasticRSI, MACD, DPO, RSI, CC


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


if __name__ == '__main__':
    # Load the configuration file
    with open('io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Clean the data
    cleaner = DataCleaner('BTC', 'io/config.yaml')
    cleaner.clean_data()
