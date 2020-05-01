import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn import init


# Create a dataloader
class GDPData(Dataset):
    """GDP time series """

    def __init__(self, csv_file, series_length=15, normalize=False):
        df = pd.read_csv(csv_file)
        self.dates =  df['Unnamed: 0'].to_numpy()
        df = df.drop(['Unnamed: 0'], axis=1)
        if normalize:
            self.normalize_table = np.zeros((df.shape[1], 2))  # (min, max)
            self.normalize(df)

        df.drop(df[(df.shape[0] // series_length *
                    series_length):].index, inplace=True)
        x = df.loc[:, (df.columns != 'GDP')]

        full = np.array(
            np.array_split(
                df.to_numpy(),
                df.shape[0] //
                series_length))
        x = np.array(np.array_split(x.to_numpy(), x.shape[0] // series_length))
        y = df["GDP"]
        y = np.array(np.array_split(y.to_numpy(), y.shape[0] // series_length))
        self.t = x.shape[0]
        x = np.expand_dims(x, -1)
        full = np.expand_dims(full, -1)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.full = torch.from_numpy(full).float()

    def normalize(self, x):
        """
          Normalize data to [-1,1] range
          for all predictors in dataframe
        """
        if not hasattr(self, 'normalize_table'):
            raise Exception("Incorrectly calling normalize")
        for idx, c in enumerate(x.columns):
            l, m = x[c].min(), x[c].max()
            self.normalize_table[idx][0] = l
            self.normalize_table[idx][1] = m
            x[c] = 2 * ((x[c] - l) / (m - l)) - 1


    def unnormalize(self, x):
        """
          Unnormalize data from [-1,1] range
          for all predictors in dataframe
        """
        if not hasattr(self, 'normalize_table'):
            raise Exception("Incorrectly calling normalize")
        for idx, c in enumerate(x.columns):
            _min = self.normalize_table[idx][0]
            _max = self.normalize_table[idx][1]
            x[c] = 0.5* (x*_max - x*_min + _max + _min)
        return x      


    def __len__(self):
        return self.t

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xr = self.x[idx]
        yr = self.y[idx]
        # the conditional
        full = self.full[idx]
        return xr, yr, full


class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()
        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size
        self.h, self.c = None, None

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
            #self.c = torch.zeros(state_size)
            #self.h = torch.zeros(state_size)
        self.h, self.c = self.lstm(x, (self.h, self.c))
        return self.h

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if not train:
            return x

        if (self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)
        return mask * x


class Generator_RNN(nn.Module):
    def __init__(
            self,
            predictor_dim,
            seq_len=15,
            target_size=1,
            hidden_dim=1,
            num_layers=1,
            dropout_prob=0,
            train=True):
        super(Generator_RNN, self).__init__()
        self.train_dp = train
        self.dropout_prob = dropout_prob
        self.lstm_basic = nn.LSTMCell(predictor_dim, hidden_dim)
        self.bn_basic = nn.BatchNorm1d(hidden_dim)

        self.dropout1 = LockedDropout()
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_dim, target_size), nn.Tanh())

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, inputx, inputc):
        #self.reset_state()
        # input - batch_size x time_steps x features
        input = torch.cat((inputx, inputc), 2)
        batch_size, no_of_timesteps, features = input.size(
            0), input.size(1), input.size(2)
        outputs = []
        # lstm on each sequence
        for i in range(batch_size):
            h,_ = self.lstm_basic(input[i, :, :])
            h = self.dropout1(h,dropout=self.dropout_prob, train=self.train_dp)
            h = self.fc_output(h)
            h = torch.squeeze(h)
            outputs.append(h)

        outputs = torch.stack(outputs)  # batch_size, timesteps
        return outputs


class Discriminator_RNN(nn.Module):
    def __init__(
            self,
            predictor_dim,
            target_size=1,
            hidden_dim=200,
            num_layers=1,
            dropout_prob=0,
            train=True):
        super(Discriminator_RNN, self).__init__()
        self.train_dp = train
        self.dropout_prob = dropout_prob
        self.lstm_basic = nn.LSTMCell(predictor_dim, hidden_dim)
        self.bn_basic = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = LockedDropout()
        self.fc_output = nn.Sequential(
            nn.Linear(
                hidden_dim,
                target_size),
            nn.Sigmoid())

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, _inputx, _inputy):
        #self.reset_state()
        # input - batch_size x time_steps x features
        _input = torch.cat((_inputx, _inputy), 2)
        no_of_timesteps = _input.shape[1]
        outputs = []

        for i in range(no_of_timesteps):
            h,_ = self.lstm_basic(_input[:, i, :])
            #h = self.dropout1(h)
            outputs.append(h)

        outputs = torch.stack(outputs)  # time_steps, batch_size, features
        outputs = outputs.permute(1, 2, 0)  # time_steps, features, batch_size

        pool = nn.MaxPool1d(no_of_timesteps)

        h = pool(outputs)
        h = h.view(h.size(0), -1)
        h = self.fc_output(h)
        return h
