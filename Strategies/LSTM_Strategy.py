import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import yaml


class LSTM:
    def __init__(self, df):
        self.df = df
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    # Load the configuration file
    with open('io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_path = config['Strategy']['LSTM']['data_path']

    df = pd.read_csv(data_path)
