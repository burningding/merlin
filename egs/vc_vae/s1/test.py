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
from src.utils.pitch import build_pitch_model, pitch_conversion
import soundfile as sf
import os
from src.dataset.dataset_utils import *
import numpy as np


def test(model, device, test_dataset, args):
    model.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            feat, label, orig_feat, result_path = test_dataset[i]
            feat = torch.from_numpy(feat)
            label = label.repeat(feat.shape[0], 0)
            label = torch.from_numpy(label)
            feat = feat.to(device)
            label = label.to(device)
            _, _, conv_feat, _ = model(feat, label)
            conv_feat = conv_feat.cpu().numpy()
            conv_feat = np.hstack([orig_feat[:, 0].reshape(orig_feat.shape[0], 1), conv_feat])
            conv_feat = test_dataset.undo_mvn(conv_feat)
            write_binfile(conv_feat, result_path)



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
    check_point = torch.load(os.path.join(args.model_dir, 'model_18.pth'))

    test_dataset = Arctic('test', feature_type=args.feature_type, speakers=['bdl', 'rms'], src_speaker='bdl', tgt_speaker='bdl',
                          utt_index=[i for i in range(251, 301)],  transform=None, use_mvn=True)

    model = VaeMlp().to(device)
    model.load_state_dict(check_point['state_dict'])

    test(model, device, test_dataset, args)
