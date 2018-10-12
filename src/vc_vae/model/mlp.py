from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from .layers import FcGaussianSample
from .loss import log_gauss
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
        label = torch.squeeze(label)
        z = torch.cat((z, label), dim=1)
        return z

    def decode(self, z):
        h = torch.tanh(self.fc_dec(z))
        x_mu, x_logvar, x = self.fc_dec_gaussian_sample(h)
        px = [x_mu, x_logvar]
        return px, x

    def forward(self, x, label):
        mu, logvar, z = self.encode(x.view(-1, input_dim))
        z = self.condition_label(z, label)
        px, x = self.decode(z)
        return px, x, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(px, x, mu, logvar):
    N = x.shape[0]
    x = torch.squeeze(x)
    rec_mu, rec_logvar = px
    logpx_z = 0.5 * torch.sum(log_gauss(rec_mu, rec_logvar, x))
    # logpx_z = F.mse_loss(rec_mu, x, size_average=False)
    # logpx_z = torch.sum((x - rec_mu).pow(2))
    # print('logpx_z: {}'.format(logpx_z))
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print('kld: {}'.format(kld))
    # return logpx_z + kld
    return logpx_z + kld
