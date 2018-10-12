from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


class FcGaussianSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FcGaussianSample, self).__init__()
        self.fc_mu = nn.Linear(input_dim, output_dim)
        self.fc_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mu)
        return mu, logvar, sample