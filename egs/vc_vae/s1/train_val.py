from __future__ import print_function
import argparse
import torch
from torch import optim
from torch.nn import functional as F
from src.model.mlp import VaeMlp, vae_loss_function
from src.dataset.Arctic import Arctic
from torch.utils.data import DataLoader
from src.dataset.transform import Compose, LogTransform, Normalization
from src.utils.model_io import save_checkpoint
from src.utils.pitch import *
from src.utils.logger import *
import datetime



def train(epoch, model, device, optimizer, train_loader, args):
    model.train()
    train_loss = 0
    for batch_idx, sampled_batch in enumerate(train_loader):
        data, label = sampled_batch
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        px, x, mu, logvar = model(data, label)
        loss = vae_loss_function(px, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    info('====> Epoch: {} Average loss: {:.4f}'.format(
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
            label = label.to(device)
            px, x, mu, logvar = model(data, label)
            test_loss += vae_loss_function(px, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    info('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE for VC')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-dir', type=str, default='./exp/arctic/log', metavar='L',
                        help='the directory of the log')
    parser.add_argument('--dataset', type=str, default='arctic', metavar='D',
                        help='the dataset used for the experiment')
    parser.add_argument('--feature_type', type=str, default='mgc', metavar='F',
                        help='the feature type used for the experiment')

    # VC speakers
    parser.add_argument('--speakers', type=str, default='bdl slt', metavar='S1S2SN',
                        help='All the speakers in training')
    # utterance indices
    parser.add_argument('--num-train-utt', type=int, default=250, metavar='T',
                        help='number of training utterances')
    parser.add_argument('--num-val-utt', type=int, default=50, metavar='V',
                        help='number of validation utterances')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    speakers = args.speakers.split()

    if args.dataset == 'arctic':
        args.model_dir = './exp/arctic/model/model_' + '_'.join(speakers)
        args.name_format = 'arctic_a{0:04d}'
        args.pitch_dir = './exp/arctic/pitch_model'

    set_logger(custom_logger("{0}/train_{1}.log".format(args.log_dir, datetime.datetime.now())))

    train_loader = DataLoader(
        Arctic('train', feature_type=args.feature_type, speakers=speakers, utt_index=[i for i in range(1, 1 + args.num_train_utt)],
               transform=None, use_mvn=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        Arctic('val', feature_type=args.feature_type, speakers=speakers, utt_index=[i for i in range(1 + args.num_train_utt,
                                                                                                           1 + args.num_train_utt + args.num_val_utt)],
               transform=None, use_mvn=True),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    args.feature_path = train_loader.dataset._feature_path
    build_pitch_model(speakers, args)

    model = VaeMlp().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, device, optimizer, train_loader, args)
        test(epoch, model, device, test_loader)
