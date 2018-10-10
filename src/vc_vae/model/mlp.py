from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from conf.pytorch.config import num_speakers

input_dim = 59
hidden_dim = 64
latent_dim = 32

class VaeMlp(nn.Module):
    def __init__(self):
        super(VaeMlp, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim + num_speakers, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        h2 = F.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def condition_label(self, z, label):
        label = torch.squeeze(label)
        z = torch.cat((z, label), dim=1)
        return z

    def decode(self, z):
        h4 = F.tanh(self.fc4(z))
        h5 = F.tanh(self.fc5(h4))
        return F.sigmoid(self.fc6(h5))

    def forward(self, x, label):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        z = self.condition_label(z, label)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD