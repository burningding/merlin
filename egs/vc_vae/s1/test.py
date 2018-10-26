from __future__ import print_function
import argparse
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.model.mlp import VaeMlp
from src.model.lstm import VaeLstm
from src.model.loss import vae_loss_function
from src.dataset.Arctic import Arctic
from src.dataset.data_loader import FrameDataLoader
from src.dataset.transform import Compose, LogTransform, Normalization
from src.utils.model_io import save_checkpoint
from src.utils.pitch import build_pitch_model, pitch_conversion
import soundfile as sf
import os
from src.dataset.dataset_utils import *
import numpy as np
import datetime
from src.utils.pitch import *
from src.utils.logger import *

def test(model, device, test_loader, args):
    model.eval()
    conv_feats = []
    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(test_loader):
            data, label = sampled_batch
            data = data.to(device)
            label = label.to(device)
            if args.model == 'lstm':
                batch_size = data.shape[0]
                model.hidden_enc = model.init_hidden(batch_size)
                model.hidden_dec = model.init_hidden(batch_size)
                data = data.permute(1, 0, 2)
            px, _, _, _ = model(data, label)
            conv_feat = px[0]
            conv_feat = conv_feat.permute(1, 0, 2)
            conv_feat = conv_feat.cpu().numpy()
            conv_feats.append(conv_feat)
    conv_feats = np.vstack(conv_feats)
    return conv_feats

# def test(model, device, test_loader, args):
#     model.eval()
#     src_pitch_model = load_pitch_model(os.path.join(args.pitch_dir, args.src_speaker + '.pkl'))
#     tgt_pitch_model = load_pitch_model(os.path.join(args.pitch_dir, args.tgt_speaker + '.pkl'))
#     with torch.no_grad():
#         for batch_idx, sampled_batch in enumerate(test_loader):
#             sample = test_dataset[i]
#             convert_feature(model, sample, device, args)
#             convert_pitch(sample, src_pitch_model, tgt_pitch_model)

def voice_conversion(conv_feats, test_dataset, args):
    src_pitch_model = load_pitch_model(os.path.join(args.pitch_dir, args.src_speaker + '.pkl'))
    tgt_pitch_model = load_pitch_model(os.path.join(args.pitch_dir, args.tgt_speaker + '.pkl'))
    speech_db = test_dataset.get_speech_db()
    input_len = test_dataset.get_input_len()
    feature_dim = test_dataset.get_feature_dim()
    feature_type = test_dataset.get_feature_type()
    idx_count = 0
    for speech_dict in speech_db:
        utt_len = speech_dict['utt_len']
        num_seq = utt_len // input_len
        if num_seq * input_len < utt_len:
            num_seq += 1
        feat = conv_feats[idx_count: idx_count + num_seq, :, :]
        idx_count += num_seq
        feat = np.reshape(feat, (-1, feat.shape[-1]))
        feat = feat[:utt_len, :]
        feat = test_dataset.undo_mvn(feat)
        orig_feat = read_binfile(speech_dict['feat_path'], feature_dim)
        feat = np.hstack([orig_feat[:, 0].reshape(orig_feat.shape[0], 1), feat])
        result_path = speech_dict['result_path']
        write_binfile(feat, result_path.format(feature_type, feature_type))
        lf0 = read_binfile(speech_dict['lf0_path'], 1)
        conv_lf0 = pitch_conversion(lf0, src_pitch_model, tgt_pitch_model)
        write_binfile(conv_lf0, result_path.format('lf0', 'lf0'))


# def convert_feature(model, sample, device, args):
#     feat, label, orig_feat, result_path, lf0 = sample
#     feat = torch.from_numpy(feat)
#     label = label.repeat(feat.shape[0], 0)
#     label = torch.from_numpy(label)
#     feat = feat.to(device)
#     label = label.to(device)
#     px, _, _, _ = model(feat, label)
#     conv_feat = px[0]
#     conv_feat = conv_feat.cpu().numpy()
#     conv_feat = np.hstack([orig_feat[:, 0].reshape(orig_feat.shape[0], 1), conv_feat])
#     conv_feat = test_dataset.undo_mvn(conv_feat)
#     write_binfile(conv_feat, result_path.format(args.feature_type, args.feature_type))


# def convert_pitch(sample, src_pitch_model, tgt_pitch_model):
#     _, _, _, result_path, lf0 = sample
#     conv_lf0 = pitch_conversion(lf0, src_pitch_model, tgt_pitch_model)
#     write_binfile(conv_lf0, result_path.format('lf0', 'lf0'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VAE for VC')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-dir', type=str, default='./exp/arctic/log', metavar='L',
                        help='the directory of the log')
    parser.add_argument('--dataset', type=str, default='Arctic', metavar='D',
                        help='the dataset used for the experiment')


    parser.add_argument('--feature_type', type=str, default='mgc', metavar='F',
                        help='the feature type used for the experiment')
    parser.add_argument('--input-shift', type=int, default=20, metavar='L',
                        help='the shift of the sequence in a sentence')

    # model type
    parser.add_argument('--model', type=str, default='mlp', metavar='M',
                        help='the type of model that are using, mlp or lstm')

    # VC speakers
    parser.add_argument('--speakers', type=str, default='bdl slt', metavar='S1S2SN',
                        help='All the speakers in training')
    parser.add_argument('--src-speaker', type=str, default='bdl', metavar='S',
                        help='src speaker in VC')
    parser.add_argument('--tgt-speaker', type=str, default='slt', metavar='T',
                        help='tgt speaker in VC')

    # utterance indices
    parser.add_argument('--num-test-utt', type=int, default=50, metavar='T',
                        help='number of testing utterances')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    speakers = args.speakers.split()

    if args.dataset == 'Arctic':
        args.model_dir = './exp/arctic/model/model_' + '_'.join(speakers)
        args.name_format = 'arctic_{0}{1:04d}'
        args.pitch_dir = './exp/arctic/pitch_model'
        args.dataset_path = './dataset/arctic'
        args.exp_dir = './exp/arctic'

    set_logger(custom_logger("{0}/test_{1}.log".format(args.log_dir, datetime.datetime.now())))

    if args.model == 'mlp':
        model = VaeMlp().to(device)
        args.input_type = 'frame'
    elif args.model == 'lstm':
        model = VaeLstm(device).to(device)
        args.input_type = 'seq'

    check_point = torch.load(os.path.join(args.model_dir, 'model_20.pth'))

    model.load_state_dict(check_point['state_dict'])

    test_dataset = Arctic('test', args.dataset_path, args.exp_dir, args.name_format,
               feature_type=args.feature_type, speakers=speakers, src_speaker=args.src_speaker, tgt_speaker=args.tgt_speaker,
               utt_index=[i for i in range(251, 251 + args.num_test_utt)],
               transform=None, use_mvn=True, input_type=args.input_type, input_shift=20)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    conv_feats = test(model, device, test_loader, args)

    voice_conversion(conv_feats, test_dataset, args)



