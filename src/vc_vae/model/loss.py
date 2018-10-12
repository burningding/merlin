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