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


class TimeSeriesTransformerForecaster:

    def __init__(self,
                 embedding_size,
                 kernel_size,
                 nb_heads,
                 forward_expansion,
                 forward_expansion_decoder,
                 max_len,
                 fc_out,
                 dropout,
                 normalize,
                 path_weight_save,
                 path_loss_save,
                 device='cuda:0'):

        super(TimeSeriesTransformerForecaster, self).__init__()

        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.nb_heads = nb_heads
        self.forward_expansion = forward_expansion
        self.forward_expansion_decoder = forward_expansion_decoder
        self.max_len = max_len
        self.fc_out = fc_out
        self.dropout = dropout
        self.normalize = normalize
        self.device = device

        self.path_weight_save = path_weight_save
        self.path_loss_save = path_loss_save

        # Convolution for signal embedding
        self.embed_conv = nn.Conv1d(in_channels=1,
                                    out_channels=self.embedding_size,
                                    kernel_size=self.kernel_size,
                                    stride=1,
                                    padding=5,
                                    bias=False)

        # Instance normalization
        self.istance_norm = nn.InstanceNorm2d(num_features=self.embedding_size, affine=True)
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.embedding_size, self.max_len)
        # Dropout on embedding and positional encoding
        self.dropout_layer = nn.Dropout(self.dropout)

        self.bk1 = TransformerBlock(self.embedding_size, self.nb_heads,
                                    self.forward_expansion, self.dropout)
        self.bk2 = TransformerBlock(self.embedding_size, self.nb_heads,
                                    self.forward_expansion, self.dropout)
        self.bk3 = TransformerBlock(self.embedding_size, self.nb_heads,
                                    self.forward_expansion, self.dropout)
        self.bk4 = TransformerBlock(self.embedding_size, self.nb_heads,
                                    self.forward_expansion, self.dropout)

        self.dec_fc1 = nn.Linear(self.embedding_size, self.embedding_size * self.forward_expansion_decoder)
        self.re1 = nn.ReLU()

        self.dec_fc2 = nn.Linear(self.embedding_size * self.forward_expansion_decoder,
                                 self.embedding_size * self.forward_expansion_decoder)
        self.re2 = nn.ReLU()

        self.dec_fc3 = nn.Linear(self.embedding_size * self.forward_expansion_decoder, self.embedding_size)
        self.re3 = nn.ReLU()

        self.dec_fc4 = nn.Linear(self.embedding_size, self.fc_out)

        # Softmax for the end
        self.sm = nn.Softmax(dim=2)

    def forward(self, x):

        # Normalization
        if self.normalize:
            x = nn.functional.normalize(x, dim=1)

        beat_size = x.size(1)
        # Adapt the shape for conv1D
        x = torch.reshape(x, (x.size(0), 1, x.size(1)))
        # Build embedding form of signals using conv layer
        out = self.embed_conv(x)
        out = out[:, :, 0:beat_size]

        # Add the positional encoding
        out = self.pos_encoding(out)

        # Transpose last two dim to have embedding on the last dim
        out.transpose_(1, 2)

        # First encoders
        out = self.bk1(out, out, out, mask=None)
        out = self.bk2(out, out, out, mask=None)
        out = self.bk3(out, out, out, mask=None)
        out = self.bk4(out, out, out, mask=None)

        # Decoder part
        out = self.re1(self.dec_fc1(out))
        out = self.re2(self.dec_fc2(out))
        out = self.re3(self.dec_fc3(out))
        out = torch.sigmoid(self.dec_fc4(out))

        return out


class SelfAttention(nn.Module):
    """
    The self attention module
    """
    def __init__(self, embed_size=5, nb_heads=5):
        """
        :param embed_size: This size is the kernel size of the embedding
        convolutional layer.
        :param nb_heads: The number of heads in the self attention process
        """
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        # WARNING: the embed_size have to be a multiple of the number of heads
        self.nb_heads = nb_heads
        self.heads_dim = int(embed_size / nb_heads)

        # Layer to generate the values matrix
        self.fc_values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Layer to generate keys
        self.fc_keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        # Layer for Queries
        self.fc_queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)

        # A fully connected layer to concatenate results
        self.fc_concat = nn.Linear(self.nb_heads * self.heads_dim, embed_size)

        # The softmax step
        self.sm = nn.Softmax(dim=3)

    def forward(self, values, keys, query, mask=None):

        # Get the number of training samples
        n = query.shape[0]
        # Get original shapes
        v_len = values.size()[1]
        k_len = keys.size()[1]
        q_len = keys.size()[1]

        # Split embedded inputs into the number of heads
        v = values.view(n, v_len, self.nb_heads, self.heads_dim)
        k = keys.view(n, k_len, self.nb_heads, self.heads_dim)
        q = query.view(n, q_len, self.nb_heads, self.heads_dim)

        # Feed it in appropriate layer
        v = self.fc_values(v)
        k = self.fc_keys(k)
        q = self.fc_queries(q)

        # Matrix dot product between queries and keys
        prdc = torch.einsum('nqhd,nkhd->nhqk', [q, k])

        # Apply mask if present
        if mask is not None:
            prdc = prdc.masked_fill(mask == 0, float('-1e20'))  # don't use zero

        # The softmax step
        #attention = self.sm(prdc / (self.embed_size ** (1/2)))
        attention = torch.softmax(prdc / (self.embed_size ** (1 / 2)), dim=3)

        # Product with values
        # Output shape: (n, query len, heads, head_dim
        out = torch.einsum('nhql,nlhd->nqhd', [attention, v])

        # Concatenate heads results (n x query len x embed_size)
        out = torch.reshape(out, (n, q_len, self.nb_heads * self.heads_dim))

        # Feed the last layer
        return self.fc_concat(out)


class TransformerBlock(nn.Module):

    def __init__(self, embed_size=50, nb_heads=5, forward_expansion=4, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # The self attention element
        self.attention = SelfAttention(embed_size, nb_heads)

        # The first normalization after attention block
        self.norm_A = nn.LayerNorm(embed_size)

        # The feed forward part
        self.feed = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        # The second normalization
        self.norm_B = nn.LayerNorm(embed_size)

        # A batch normalization instead of the classical dropout
        #self.bn = nn.BatchNorm1d(embed_size)

        # Or a classical dropout to avoid overfitting
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v, k, q, mask=None):

        # The attention process
        out = self.attention(v, k, q, mask)

        # The first normalization after attention + skip connection
        out = self.norm_A(out + q)

        # A batch normalization instead of the classical dropout
        #out = self.bn(out)

        # The feed forward part
        fw = self.feed(out)

        # The second normalization + skip connection
        out = self.norm_B(fw + out)

        # An other batch normalization
        #out = self.bn(out)

        # Dropout
        out = self.dropout(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, max_len):
        super(PositionalEncoding, self).__init__()

        self.embed_size = embed_size
        self.max_len = max_len

        # Store a matrix with all possible positions
        pe = torch.zeros(embed_size, max_len)
        for pos in range(0, max_len):
            for i in range(0, embed_size, 2):
                pe[i, pos] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[i+1, pos] = math.cos(pos / (10000 ** ((2 * (i+1)) / embed_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # Get seq size
        seq_len = x.size(2)

        # If size is greater that pos embedding saved in memory:
        if seq_len > self.max_len:
            self.adapt_len(seq_len)
        # Add positional embedding
        x = x[:, 0:self.embed_size, 0:seq_len] + self.pe[0, :, :seq_len].to('cuda:0')
        return x

if __name__ == '__main__':
    # Load the configuration file
    with open('../io/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the torch data
    X = torch.load('../' + config['Strategy']['Transformers']['data_path_X'])
    y = torch.load('../' + config['Strategy']['Transformers']['data_path_y'])

    save_path = '../' + config['Strategy']['Transformers']['weights_path']

    LOAD_WEIGHTS = config['Strategy']['Transformers']['load_weights']
    BATCH_SIZE = config['Strategy']['Transformers']['batch_size']
    LEARNING_RATE = config['Strategy']['Transformers']['learning_rate']
    NB_EPOCHS = config['Strategy']['Transformers']['nb_epochs']

    EMBEDDING_SIZE = config['Strategy']['Transformers']['embedding_size']
    KERNEL_SIZE = config['Strategy']['Transformers']['kernel_size']
    NB_HEADS = config['Strategy']['Transformers']['nb_heads']
    FORWARD_EXPANSION = config['Strategy']['Transformers']['forward_expansion']
    FORWARD_EXPANSION_DECODER = config['Strategy']['Transformers']['forward_expansion_decoder']
    MAX_LEN = config['Strategy']['Transformers']['max_len']

    DROPOUT = config['Strategy']['Transformers']['dropout']
    NORMALIZE = config['Strategy']['Transformers']['normalize']

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = f"Transformers_ES_{EMBEDDING_SIZE}_KS_{KERNEL_SIZE}_NH_{NB_HEADS}_FE_{FORWARD_EXPANSION}"

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.unsqueeze(2)
    X_test = X_test.unsqueeze(2)

    model = TimeSeriesTransformerForecaster(
        embedding_size=EMBEDDING_SIZE,
        kernel_size=KERNEL_SIZE,
        nb_heads=NB_HEADS,
        forward_expansion=FORWARD_EXPANSION,
        forward_expansion_decoder=FORWARD_EXPANSION_DECODER,
        max_len=MAX_LEN,
        fc_out=y_train.shape[1],
        dropout=DROPOUT,
        normalize=NORMALIZE,
        path_weight_save=save_path,
        path_loss_save=save_path,
        device=DEVICE
    ).to(DEVICE)

    if LOAD_WEIGHTS:
        try:
            model.load_state_dict(torch.load(save_path + MODEL_NAME + ".pt"))
        except:
            print("No weights found")

    train_loader = DataLoader(TimeSeriesDataframe(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataframe(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_train_loss = np.zeros((NB_EPOCHS, 1))
    epoch_test_loss = np.zeros((NB_EPOCHS, 1))

    for i in tqdm(range(0, NB_EPOCHS)):

        tmp_train_loss = np.zeros((len(train_loader), 1))
        tmp_test_loss = np.zeros((len(test_loader), 1))

        for idx, (signal, target) in enumerate(train_loader):

            model.train()

            signal = signal.to(DEVICE)
            target = target.to(DEVICE)

            # Reset grad
            optimizer.zero_grad()
            # Make predictions
            preds = model(signal.to(DEVICE))

            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

            tmp_test_loss_ = np.zeros((len(test_loader), 1))

        model.eval()
        with torch.no_grad():
            for idx, (signal, target) in enumerate(test_loader):
                signal = signal.to(DEVICE)
                target = target.to(DEVICE)

                preds = model(signal.to(DEVICE))

                loss = criterion(preds, target)

                tmp_test_loss[idx] = np.mean(loss.cpu().detach().item())

        epoch_train_loss[i] = np.mean(tmp_train_loss)
        epoch_test_loss[i] = np.mean(tmp_test_loss)

    # Save the model
    torch.save(model.state_dict(), save_path + MODEL_NAME + ".pt")

    # Plot the loss
    plt.plot(epoch_train_loss, label='Training Loss')
    plt.plot(epoch_test_loss, label='Test Loss')
    plt.legend()
    plt.show()