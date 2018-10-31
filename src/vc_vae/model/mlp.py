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

class VaeMlp(nn.Module):
    def __init__(self):
        super(VaeMlp, self).__init__()

        self.fc_enc = nn.Linear(input_dim, hidden_dim)
        self.fc_enc_gaussian_sample = FcGaussianSample(hidden_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim + num_speakers, hidden_dim)
        # self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec_gaussian_sample = FcGaussianSample(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.tanh(self.fc_enc(x))
        mu, logvar, z = self.fc_enc_gaussian_sample(h)
        return mu, logvar, z

    def condition_label(self, z, label):
        label = label.view(z.shape[0], -1)
        z = torch.cat((z, label), dim=1)
        return z

    def decode(self, z):
        h = torch.tanh(self.fc_dec(z))
        x_mu, x_logvar, x = self.fc_dec_gaussian_sample(h)
        px = [x_mu, x_logvar]
        return px, x

    def forward(self, x, label):
        x = x.view(-1, input_dim)
        mu, logvar, z = self.encode(x)
        z = self.condition_label(z, label)
        px, x = self.decode(z)
        return px, x, mu, logvar



