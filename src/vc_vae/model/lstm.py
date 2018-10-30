from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from .layers import FcGaussianSample
from conf.pytorch.config import num_speakers

input_dim = 59
hidden_dim = 32
latent_dim = 16
seq_len = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VaeLstm(nn.Module):
    def __init__(self):
        super(VaeLstm, self).__init__()
        self.lstm_enc = nn.LSTM(input_dim, hidden_dim)
        self.fc_enc_gaussian_sample = FcGaussianSample(hidden_dim, latent_dim)
        self.lstm_dec = nn.LSTM(latent_dim + num_speakers, hidden_dim)
        # self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec_gaussian_sample = FcGaussianSample(hidden_dim, input_dim)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, hidden_dim).to(device),
                torch.zeros(1, batch_size, hidden_dim).to(device))

    def encode(self, x):
        lstm_out, _ = self.lstm_enc(x.view(seq_len, -1, input_dim), self.hidden_enc)
        h = lstm_out[-1]
        mu, logvar, z = self.fc_enc_gaussian_sample(h)
        return mu, logvar, z

    def condition_label(self, z, label):
        label = label.view(z.shape[0], -1)
        z = torch.cat((z, label), dim=1)
        return z

    def decode(self, z):
        z = z.repeat(seq_len, 1, 1)
        lstm_out, _ = self.lstm_dec(z.view(seq_len, -1, latent_dim + num_speakers), self.hidden_dec)
        h = lstm_out
        x_mu, x_logvar, x = self.fc_dec_gaussian_sample(h)
        px = [x_mu, x_logvar]
        return px, x

    def forward(self, x, label):
        mu, logvar, z = self.encode(x)
        z = self.condition_label(z, label)
        px, x = self.decode(z)
        return px, x, mu, logvar



