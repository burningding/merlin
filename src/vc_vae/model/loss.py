import torch
import numpy as np

def log_gauss(mu, logvar, x):
    """compute point-wise log prob of Gaussian"""
    # const = torch.log(torch.tensor(2 * np.pi, device=torch.device('cuda')))
    # res = -0.5 * (const + logvar + (x - mu).pow(2) / logvar.exp())
    # res = 0.798 + logvar + (x - mu).pow(2) / logvar.exp()
    # print 'logvar: {}'.format(torch.mean(logvar).data[0])
    # print 'mu: {}'.format(torch.mean(mu).data[0])
    assert mu.shape == logvar.shape, 'Shape mismatch between input 0 and input 1'
    assert mu.shape == x.shape, 'Shape mismatch between input 0 and input 2'
    return 0.798 + logvar + (x - mu).pow(2) / logvar.exp()

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(px, x, mu, logvar):
    # N = x.shape[0]
    x = torch.squeeze(x)
    rec_mu, rec_logvar = px
    logpx_z = 0.5 * torch.sum(log_gauss(rec_mu, rec_logvar, x))
    # if len(x.shape) > 2:
    #     logpx_z /= x.shape[0]
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