import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import torch
import torch.nn as nn

import math

import yaml

from tqdm import tqdm

from utils.trainer import Trainer
from utils.utils import train_test_split_custom

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from utils.utils import print_size, print_underlined

# Load the configuration file
with open('../io/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# ------------------------------------------------------------------------ #
#                             Load the parameters                          #
# ------------------------------------------------------------------------ #

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
PADDING = config['Strategy']['Transformers']['padding']

DROPOUT = config['Strategy']['Transformers']['dropout']
NORMALIZE = config['Strategy']['Transformers']['normalize']

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_NAME = f"Transformers_ES_{EMBEDDING_SIZE}_NH_{NB_HEADS}_FE_{FORWARD_EXPANSION}_NBL_{NB_LAYERS}"


class TimeSeriesDataframe(Dataset):
    """
    A class to create a time series dataframe.
    """

    def __init__(self, X, y):
        super().__init__()

        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeSeriesTransformerForecaster(nn.Module):
    """
    A class to create a transformer model for time series forecasting.

    1. nn.ConvTranspose1d
    ------------

    Convolution for  embedding
    [batch_size, in_channels, signal_length] -> [batch_size, out_channels, signal_length_out]
    with L_out = (L_in - 1) * stride - 2 * padding + kernel_size (dilation = 1, output_padding = 0)
    """

    def __init__(self,
                 n_features,
                 embedding_size=200,
                 num_layers=8,
                 expansion=4,
                 dropout=0.1,
                 kernel_size=5,
                 padding=25,
                 nhead=10,
                 debug=False):
        super(TimeSeriesTransformerForecaster, self).__init__()

        assert embedding_size % n_features == 0, "The embedding size must be a multiple of the number of features."

        self.DROPOUT = dropout
        self.N_FEATURES = n_features
        self.NHEAD = nhead
        self.NUM_LAYERS = num_layers
        self.DEBUG = debug
        self.KERNEL_SIZE = kernel_size
        self.STRIDE = 1
        self.EMBEDDING_SIZE = embedding_size
        self.PADDING = padding
        self.EXPANSION = expansion

        # 1. Embedding
        self.embedding = EmbeddingLayer(
            n_features=n_features,
            embedding_size=embedding_size,
            type_="duplicate"
        )

        self.pos_encoder = PositionalEncoding(d_model=self.EMBEDDING_SIZE, dropout=self.DROPOUT, debug=self.DEBUG)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.EMBEDDING_SIZE, nhead=self.NHEAD, dropout=self.DROPOUT),
            num_layers=self.NUM_LAYERS
        )

        self.fc_decoder_1 = nn.Linear(self.EMBEDDING_SIZE, self.EMBEDDING_SIZE * self.EXPANSION)
        self.re1 = nn.ReLU()

        self.fc_decoder_2 = nn.Linear(self.EMBEDDING_SIZE * self.EXPANSION, self.EMBEDDING_SIZE)
        self.re2 = nn.ReLU()

        self.fc_decoder_3 = nn.Linear(self.EMBEDDING_SIZE, 1)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.fc_decoder_1.bias.data.zero_()
        self.fc_decoder_1.weight.data.uniform_(-init_range, init_range)

        self.fc_decoder_2.bias.data.zero_()
        self.fc_decoder_2.weight.data.uniform_(-init_range, init_range)

        self.fc_decoder_3.bias.data.zero_()
        self.fc_decoder_3.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        l_out = (x.size(0) - 1) * self.STRIDE - 2 * self.PADDING + self.KERNEL_SIZE

        assert l_out == x.size(0), "L_out must match the input signal size."

        if self.DEBUG:
            print_underlined("Time series transformer")
            print_size("x", x)
        x = self.embedding(x)

        if self.DEBUG:
            print_size("x_deconv", x)

        x = x.permute(2, 0, 1)

        x = self.pos_encoder(x)

        if self.DEBUG:
            print_size("x_permuted and pos_encoded", x)

        x = self.transformer_encoder(x)

        if self.DEBUG:
            print_size("x_transformer_encoded", x)

        # x = self.re1(self.fc_decoder_1(x))
        #
        # x = self.re2(self.fc_decoder_2(x))

        x = self.fc_decoder_3(x)

        if self.DEBUG:
            print_size("x_fc_decoded", x)

        x = x.permute(1, 2, 0)

        if self.DEBUG:
            print_size("x_permuted", x)

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
            print_size("pe_resized", self.pe[:x.size(0), :, :])
            print_size("x + pe_resized", x + self.pe[:x.size(0), :, :])

        return self.dropout(x + self.pe[:x.size(0), :, :])


class EmbeddingLayer(nn.Module):
    def __init__(self, n_features, embedding_size, type_):
        super(EmbeddingLayer, self).__init__()

        self.N_FEATURES = n_features
        self.EMBEDDING_SIZE = embedding_size
        self.TYPE = type

        kernel_size = 5
        padding = 2
        stride = 1

        self.embedding_conv = nn.ConvTranspose1d(in_channels=n_features,
                                                 out_channels=embedding_size,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 bias=False)

    def forward(self, x):

        if self.TYPE == "deconv":
            x = self.embedding_conv(x)
        elif self.TYPE == "duplicate":
            # If only prices, the feature size is 1, we duplicate it embedding_size times
            # If n_features different from 1, we duplicate the features so that ot matches the embedding size
            if x.size(1) == 1:
                x = x.repeat(1, self.EMBEDDING_SIZE, 1)
            else:
                x = x.repeat(1, self.EMBEDDING_SIZE / x.size(1), 1)
        return x



if __name__ == '__main__':

    # ------------------------------------------------------------------------ #
    #                               Load the data                              #
    # ------------------------------------------------------------------------ #

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

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

    # ------------------------------------------------------------------------ #
    #                               Define training                            #
    # ------------------------------------------------------------------------ #

    model = TimeSeriesTransformerForecaster(
        n_features=X_train.size(1),
        embedding_size=EMBEDDING_SIZE,
        num_layers=NB_LAYERS,
        dropout=DROPOUT,
        nhead=NB_HEADS,
        expansion=FORWARD_EXPANSION,
        debug=DEBUG,
        padding=PADDING,
        kernel_size=KERNEL_SIZE
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

    # ------------------------------------------------------------------------ #
    #                              Train and test                              #
    # ------------------------------------------------------------------------ #

    trainer = Trainer(train_loader, test_loader, model, optimizer, criterion, DEVICE, NB_EPOCHS,
                      save_path_loss, save_path_weights, MODEL_NAME, DEBUG)

    trainer.train()

    # # Load the loss
    # train_loss_ = np.load("io/Transformers/Losses/Transformers_ES_200_NH_10_FE_4_NBL_8_train_loss.npy")
    # test_loss_ = np.load("io/Transformers/Losses/Transformers_ES_200_NH_10_FE_4_NBL_8_test_loss.npy")
    # # Plot the loss
    # plt.plot(test_loss_)
    # plt.plot(train_loss_)
    # plt.show()
