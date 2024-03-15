import pandas as pd
import numpy as np

import yaml
import os

from tqdm import tqdm

from utils.technical_analysis import StochasticRSI
from utils.technical_analysis import MACD
from utils.technical_analysis import DPO
from utils.technical_analysis import RSI
from utils.technical_analysis import CC

# Import yaml file
with open('io/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

to_load = 'ETH'

# list of the files
files = os.listdir(config['data_path'][f'{to_load}']['1min']['raw'])

for file in tqdm(files):

    # Load the data
    data = pd.read_csv(config['data_path'][f'{to_load}']['1min']['raw'] + file)

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
    data.to_csv(config['data_path'][f'{to_load}']['1min']['cleaned'] + file, index=False)
