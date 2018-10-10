from __future__ import print_function
import argparse
import torch
from torch import optim
from torch.nn import functional as F
from src.model.mlp import VaeMlp, vae_loss_function
from src.dataset.Arctic import Arctic
from src.dataset.data_loader import FrameDataLoader
from src.dataset.transform import Compose, LogTransform, Normalization
from src.utils.model_io import save_checkpoint


def train(epoch, model, device, optimizer, train_loader, args):
    model.train()
    train_loss = 0
    for batch_idx, sampled_batch in enumerate(train_loader):
        data, label = sampled_batch
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 5, len(train_loader.dataset),
                100. * batch_idx * 5. / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, args.model_dir, 'model_{}.pth'.format(epoch))


def test(epoch, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data, label)
            test_loss += vae_loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE for VC')
    parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='Arctic', metavar='D',
                        help='the dataset used for the experiment')
    parser.add_argument('--feature_type', type=str, default='mgc', metavar='F',
                        help='the feature type used for the experiment')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    if args.dataset == 'Arctic':
        args.model_dir = './exp/arctic/model'

    train_loader = FrameDataLoader(
        Arctic('train', feature_type=args.feature_type, speakers=['bdl', 'slt'], utt_index=[i for i in range(1, 101)],
               transform=Compose([Normalization(True)])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = FrameDataLoader(
        Arctic('test', feature_type=args.feature_type, speakers=['bdl', 'rms'], utt_index=[i for i in range(500, 550)],
               transform=Compose([Normalization(False)])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = VaeMlp().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, device, optimizer, train_loader, args)
        # test(epoch, model, device, test_loader)
