"""
This script is used to create a transformer model for time series forecasting.
Based on : Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case
https://arxiv.org/abs/2001.08317
Neo Wu, Bradley Green, Xue Ben, Shawn O'Banion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import os

import torch
import torch.nn as nn

import math

import yaml

from tqdm import tqdm

from utils.trainer import Trainer
from utils.utils import train_test_split_custom

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils.utils import print_size, print_underlined, print_red, print_green, generate_square_subsequent_mask

from data_handler import DataCleaner

# Load the configuration file
with open('../io/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# ------------------------------------------------------------------------ #
#                             Load the parameters                          #
# ------------------------------------------------------------------------ #

SAVE_PATH_WEIGHTS = '../' + config['Strategy']['Transformers']['weights_path']
SAVE_PATH_LOSS = '../' + config['Strategy']['Transformers']['loss_path']
SAVE_PATH = '../' + config['Strategy']['Transformers']['save_path']

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
PADDING = config['Strategy']['Transformers']['padding']

DROPOUT = config['Strategy']['Transformers']['dropout']
NORMALIZE = config['Strategy']['Transformers']['normalize']

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_NAME = f"Transformers_ES_{EMBEDDING_SIZE}_NH_{NB_HEADS}_FE_{FORWARD_EXPANSION}_NBL_{NB_LAYERS}"

WANDB = config['Strategy']['Transformers']['wandb']

# ------------------------------------------------------------------------ #
#                    Load the params for Dataset making                    #
# ------------------------------------------------------------------------ #
WINDOW = config['Strategy']['Transformers']['dh_params']['window']
LOOK_FORWARD = config['Strategy']['Transformers']['dh_params']['look_forward']
DECODER_HORIZON = config['Strategy']['Transformers']['dh_params']['decoder_horizon']
LOG_CLOSE = config['Strategy']['Transformers']['dh_params']['log_close']
CLOSE_RETURNS = config['Strategy']['Transformers']['dh_params']['close_returns']
ONLY_CLOSE = config['Strategy']['Transformers']['dh_params']['only_close']
MIN_MAX_SCALE = config['Strategy']['Transformers']['dh_params']['min_max_scaling']


class TimeSeriesDataframe(Dataset):
    """
    A class to create a time series dataframe.
    """

    def __init__(self, X_encoder, X_decoder, y):
        super().__init__()

        self.X_encoder = X_encoder
        self.X_decoder = X_decoder
        self.y = y

    def __len__(self):
        return self.X_encoder.size(0)

    def __getitem__(self, idx):
        return self.X_encoder[idx], self.X_decoder[idx], self.y[idx]


class TimeSeriesTransformerForecaster(nn.Module):
    """
    A class to create a transformer model for time series forecasting.
    """

    def __init__(self,
                 n_features,
                 embedding_size,
                 input_sequence_length,
                 output_sequence_length,
                 num_layers=8,
                 expansion=4,
                 dropout=0.1,
                 kernel_size=5,
                 padding=25,
                 nhead=10,
                 device="cuda:0",
                 debug=False):
        super(TimeSeriesTransformerForecaster, self).__init__()

        assert embedding_size % n_features == 0, "The embedding size must be a multiple of the number of features."

        self.DROPOUT = dropout
        self.N_FEATURES = n_features
        self.INPUT_SEQUENCE_LENGTH = input_sequence_length
        self.OUTPUT_SEQUENCE_LENGTH = output_sequence_length
        self.NHEAD = nhead
        self.NUM_LAYERS = num_layers
        self.DEBUG = debug
        self.KERNEL_SIZE = kernel_size
        self.STRIDE = 1
        self.EMBEDDING_SIZE = embedding_size
        self.PADDING = padding
        self.EXPANSION = expansion
        self.DEVICE = device

        self.encoder_mask = generate_square_subsequent_mask(
            dim1=self.OUTPUT_SEQUENCE_LENGTH,
            dim2=self.INPUT_SEQUENCE_LENGTH
        ).to(self.DEVICE)

        self.target_mask = generate_square_subsequent_mask(
            dim1=self.OUTPUT_SEQUENCE_LENGTH,
            dim2=self.OUTPUT_SEQUENCE_LENGTH
        ).to(self.DEVICE)

        # 1. Embedding
        self.embedding_encoder = EmbeddingLayer(
            n_features=self.N_FEATURES,
            embedding_size=self.EMBEDDING_SIZE,
            type_="fc_expansion"
        )

        self.embedding_decoder = EmbeddingLayer(
            n_features=1,  # We only predict the closing price
            embedding_size=self.EMBEDDING_SIZE,
            type_="fc_expansion"
        )

        self.pos_encoder = PositionalEncoding(d_model=self.EMBEDDING_SIZE, dropout=self.DROPOUT, debug=self.DEBUG)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.EMBEDDING_SIZE, nhead=self.NHEAD,
                                       dropout=self.DROPOUT, batch_first=True),
            num_layers=self.NUM_LAYERS
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.EMBEDDING_SIZE, nhead=self.NHEAD,
                                       dropout=self.DROPOUT, batch_first=True),
            num_layers=self.NUM_LAYERS
        )

        self.fc_decoder_1 = nn.Linear(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE * self.EXPANSION)
        self.fc_decoder_2 = nn.Linear(self.EMBEDDING_SIZE * self.EXPANSION, self.EMBEDDING_SIZE)

        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

        self.fc_decoder = nn.Linear(self.EMBEDDING_SIZE, 1)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1

        self.fc_decoder.bias.data.zero_()
        self.fc_decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, x_encoder, x_decoder):
        # ------------------------------------------------------------------------ #
        #                              Encoder part                                #
        # ------------------------------------------------------------------------ #

        if self.DEBUG:
            print_underlined("Time series transformer")
            print_size("x_encoder", x_encoder)

        # Embedding
        x_encoder = self.embedding_encoder(x_encoder)
        if self.DEBUG:
            print_size("x_encoder permuted and embedding:", x_encoder)

        # Positional encoding
        x_encoder = self.pos_encoder(x_encoder)
        if self.DEBUG:
            print_size("x_encoder_permuted and pos_encoded", x_encoder)

        # Transformer encoder
        x_encoder = self.transformer_encoder(x_encoder)
        if self.DEBUG:
            print_size("x_encoder_transformer_encoded", x_encoder)

        # ------------------------------------------------------------------------ #
        #                              Decoder part                                #
        # ------------------------------------------------------------------------ #

        x_decoder = self.embedding_decoder(x_decoder)

        x_decoder = self.transformer_decoder(x_decoder, x_encoder,
                                             tgt_mask=self.target_mask, memory_mask=self.encoder_mask)

        x_decoder = self.fc_decoder_1(x_decoder)
        x_decoder = self.relu_1(x_decoder)

        x_decoder = self.fc_decoder_2(x_decoder)
        x_decoder = self.relu_2(x_decoder)

        x_decoder = self.fc_decoder(x_decoder)

        return x_decoder


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
            print_size("pe_resized", self.pe[:x.size(0), :, :])
            print_size("x + pe_resized", x + self.pe[:x.size(0), :, :])

        return self.dropout(x + self.pe[:x.size(0), :, :])


class EmbeddingLayer(nn.Module):
    def __init__(self, n_features, embedding_size, type_):
        super(EmbeddingLayer, self).__init__()

        self.N_FEATURES = n_features
        self.EMBEDDING_SIZE = embedding_size
        self.TYPE = type_

        kernel_size = 5
        padding = 2
        stride = 1

        self.embedding_conv = nn.ConvTranspose1d(in_channels=n_features,
                                                 out_channels=embedding_size,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=False)

        self.encoder_fc = nn.Linear(
            in_features=n_features,
            out_features=embedding_size
        )
        self.re = nn.ReLU()

        self.init_weights()

    def forward(self, x):

        if self.TYPE == "fc_expansion":
            x = self.re(self.encoder_fc(x))
        elif self.TYPE == "duplicate":
            # If only prices, the feature size is 1, we duplicate it embedding_size times
            # If n_features different from 1, we duplicate the features so that ot matches the embedding size
            if x.size(1) == 1:
                x = x.repeat(1, self.EMBEDDING_SIZE, 1)
            else:
                x = x.repeat(1, self.EMBEDDING_SIZE / x.size(1), 1)
        return x

    def init_weights(self):
        init_range = 0.1
        self.encoder_fc.bias.data.zero_()
        self.encoder_fc.weight.data.uniform_(-init_range, init_range)


if __name__ == '__main__':

    # ------------------------------------------------------------------------ #
    #                              Create the data                             #
    # ------------------------------------------------------------------------ #
    cleaner = DataCleaner('BTC', '../io/config.yaml')
    cleaner.prepare_dataframe_transformers(
        window=WINDOW,
        look_forward=LOOK_FORWARD,
        decoder_horizon=DECODER_HORIZON,
        log_close=LOG_CLOSE,
        close_returns=CLOSE_RETURNS,
        only_close=ONLY_CLOSE,
        min_max_scale=MIN_MAX_SCALE
    )

    cleaner = DataCleaner('ETH', '../io/config.yaml')
    cleaner.prepare_dataframe_transformers(
        window=WINDOW,
        look_forward=LOOK_FORWARD,
        decoder_horizon=DECODER_HORIZON,
        log_close=LOG_CLOSE,
        close_returns=CLOSE_RETURNS,
        only_close=ONLY_CLOSE,
        min_max_scale=MIN_MAX_SCALE
    )

    # ------------------------------------------------------------------------ #
    #                               Load the data                              #
    # ------------------------------------------------------------------------ #

    # Load the torch data
    X_encoder = torch.load('../' + config['Strategy']['Transformers']['data_path_X_encoder'])
    X_decoder = torch.load('../' + config['Strategy']['Transformers']['data_path_X_decoder'])
    y = torch.load('../' + config['Strategy']['Transformers']['data_path_y'])

    # Split the data
    # X_encoder -> [N_windows, seq_len, n_features]
    X_encoder_train, X_encoder_test, X_decoder_train, X_decoder_test, y_train, y_test =\
        train_test_split_custom(X_encoder, X_decoder, y, train_size=0.8)

    if DEBUG:
        # Name the dimensions
        print_underlined("\n Data recovered from the folder")

        print_size("X_encoder_train", X_encoder_train)
        print_size("X_encoder_test", X_encoder_test)
        print_size("X_decoder_train", X_decoder_train)
        print_size("X_decoder_test", X_decoder_test)
        print_size("y_train", y_train)
        print_size("y_test", y_test)

    # ------------------------------------------------------------------------ #
    #                                Define model                              #
    # ------------------------------------------------------------------------ #

    n_features = X_encoder_train.size(2)
    input_sequence_length = X_encoder_train.size(1)
    output_sequence_length = X_decoder_train.size(1)

    model = TimeSeriesTransformerForecaster(
        n_features=n_features,
        embedding_size=EMBEDDING_SIZE,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        num_layers=NB_LAYERS,
        dropout=DROPOUT,
        nhead=NB_HEADS,
        expansion=FORWARD_EXPANSION,
        debug=DEBUG,
        padding=PADDING,
        device=DEVICE,
        kernel_size=KERNEL_SIZE
    ).to(DEVICE)

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
                    if config_folder['Strategy']['Transformers']['embedding_size'] == EMBEDDING_SIZE and \
                        config_folder['Strategy']['Transformers']['nb_heads'] == NB_HEADS and \
                        config_folder['Strategy']['Transformers']['nb_encoder_layers'] == NB_LAYERS and \
                        config_folder['Strategy']['Transformers']['forward_expansion'] == FORWARD_EXPANSION and \
                        config_folder['Strategy']['Transformers']['padding'] == PADDING and \
                        config_folder['Strategy']['Transformers']['kernel_size'] == KERNEL_SIZE and \
                        config_folder['Strategy']['Transformers']['dropout'] == DROPOUT and \
                        config_folder['Strategy']['Transformers']['dh_params']['window'] == WINDOW and \
                        config_folder['Strategy']['Transformers']['dh_params']['look_forward'] == LOOK_FORWARD and \
                        config_folder['Strategy']['Transformers']['dh_params']['log_close'] == LOG_CLOSE and \
                        config_folder['Strategy']['Transformers']['dh_params']['close_returns'] == CLOSE_RETURNS and \
                        config_folder['Strategy']['Transformers']['dh_params']['only_close'] == ONLY_CLOSE:
                        # Load the weights
                        model.load_state_dict(torch.load(SAVE_PATH + folder + "/weights/weights.pt"))
                        print_green(f"Loaded weights from {folder}")
                        break
            except:
                print_red("Failed to load the weights")

    # ------------------------------------------------------------------------ #
    #                               Define training                            #
    # ------------------------------------------------------------------------ #

    # The data loader
    train_loader = DataLoader(TimeSeriesDataframe(X_encoder_train, X_decoder_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataframe(X_encoder_test, X_decoder_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1)

    # ------------------------------------------------------------------------ #
    #                              Train and test                              #
    # ------------------------------------------------------------------------ #

    trainer = Trainer(train_loader, test_loader, model, optimizer, criterion, scheduler, DEVICE, NB_EPOCHS,
                      SAVE_PATH_LOSS, SAVE_PATH_WEIGHTS, MODEL_NAME, DEBUG, SAVE_PATH, WANDB)

    # trainer.train()
    trainer.evaluate(X_encoder_test[6000:7000, :, :], X_decoder_test[6000:7000, :, :], y_test[6000:7000, :, :])

