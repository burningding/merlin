from __future__ import absolute_import

from .SpeechDB import SpeechDB
import os
import torch
import numpy as np
from random import shuffle
from .dataset_utils import *
import pickle

class Arctic(SpeechDB):
    """
    Dataset class for CMU ARCTIC dataset
    """
    def __init__(self, split, feature_type=None, feature_dim=None, speakers=None, src_speaker=None, tgt_speaker=None, 
                 utt_index=None, transform=None, input_type=None, use_mvn=False):
        """
        :param split: 'train', 'val', or 'test'
        :param feature_type: 'spectrogram', 'mfcc', or 'mcep'
        :param speakers: speakers used in the experiment
        :param src_speaker: source speaker for VC (test only)
        :param tgt_speaker: target speaker for VC (test only)
        :param utt_index: utterance indices (in filename)
        :param transform: transform to be performed on spectral feature
        """
        SpeechDB.__init__(self, split, use_mvn)
        self._dataset_path = './dataset/arctic'
        self._exp_dir = './exp/arctic'
        self._feature_type = feature_type if feature_type else 'mgc'
        self._feature_dim = feature_dim if feature_dim else 60
        self._feature_path = self._get_default_feature_path()
        self._data_path = self._get_default_wav_path()
        self._speakers = speakers if speakers else ['bdl', 'rms', 'slt', 'clb']
        self._num_speakers = len(self._speakers)
        if self._split == 'test':
            self._src_speaker = src_speaker if src_speaker else 'bdl'
            self._tgt_speaker = tgt_speaker if tgt_speaker else 'rms'
            assert self._src_speaker in self._speakers, 'source speaker should be in speakers'
            assert self._tgt_speaker in self._speakers, 'target speaker should be in speakers'
            self._result_path = self._get_default_result_path()
        self._wav_index = utt_index if utt_index else self._get_index_from_split()
        self._speech_db = self._generate_speech_db()
        self._transform = transform
        self._input_type = input_type if input_type else 'frame'
        self._input_len = self._get_input_len()
        self._input_list = self._get_input_list()
        self._use_mvn = use_mvn
        self._mvn_params_path = self._get_default_mvn_params_path()
        self._mvn_params = self._get_mvn_params()

    def __len__(self):
        if self._split == 'train' or self._split == 'val':
            return len(self._input_list)
        else:
            return len(self._speech_db)

    def speaker_wav_path(self, speaker):
        return os.path.join(self._data_path, 'wav', speaker)

    def wav_path_from_index(self, speaker, index, sep='a'):
        """
        Get the wav path according to the file index
        :param speaker:
        :param index: file index
        :param sep:
        :return:
        """
        return os.path.join(self.speaker_wav_path(speaker), 'arctic_{}{:04d}.wav'.format(sep, index))
    
    def wav_path_at(self, i):
        """
        Get the wav path of the i-th item in the dataset
        :param i: the index of the item in the dataset
        :return:
        """
        return self.wav_path_from_index(self._speech_db[i]['speaker'], self._speech_db[i]['utt_index'])

    def speaker_feature_path(self, speaker):
        speaker_feature_path = os.path.join(self._feature_path, speaker, self._feature_type)
        return speaker_feature_path

    def feat_path_from_index(self, speaker, index, sep='a'):
        return os.path.join(self.speaker_feature_path(speaker), 'arctic_{}{:04d}.{}'.format(sep, index, self._feature_type))

    def feat_path_at(self, i):
        return self.feat_path_from_index(self._speech_db[i]['speaker'], self._speech_db[i]['utt_index'])

    def speaker_result_path(self):
        speaker_result_path = os.path.join(self._result_path, '{}2{}'.format(self._src_speaker, self._tgt_speaker), self._feature_type)
        if not os.path.exists(speaker_result_path):
            os.makedirs(speaker_result_path)
        return speaker_result_path

    def result_path_from_index(self, index, sep='a'):
        return os.path.join(self.speaker_result_path(), 'arctic_{}{:04d}.mgc'.format(sep, index))

    def result_path_at(self, i):
        return self.result_path_from_index(self._speech_db[i]['utt_index'])

    def _get_default_wav_path(self):
        wav_path = os.path.join(self._dataset_path, 'wav')
        return wav_path

    def _get_default_feature_path(self):
        default_feature_path = os.path.join(self._dataset_path, 'feature')
        return default_feature_path

    def _get_default_result_path(self):
        default_result_path = os.path.join(self._exp_dir, 'rec_feature')
        if not os.path.exists(default_result_path):
            os.mkdir(default_result_path)
        return default_result_path

    def _get_default_mvn_params_path(self):
        if self._use_mvn:
            default_mvn_params_path = os.path.join(self._exp_dir, 'mvn_params.pkl')
        else:
            default_mvn_params_path = None
        return default_mvn_params_path

    def _get_index_from_split(self):
        if self._split == 'train':
            return [i for i in range(1, 501)]
        elif self._split == 'test':
            return [i for i in range(501, 594)]
        else:
            raise NotImplementedError

    def _get_input_len(self):
        input_len = 1
        if self._input_type == 'frame':
            input_len = 1
        elif self._input_type == 'seq':
            input_len = 16
        else:
            raise ValueError('Unknown input type, use frame or seq instead')
        return input_len

    def _generate_speech_db(self):
        """
        Generate speech database structure
        :return: speech_db: list of speech_dict
                        - speech_dict: dictionary of speech data
                                - wav_fname
                                - speaker
                                - utt_index
                                - feat_fname
        """
        if self._split == 'train' or self._split == 'val':
            speech_db = []
            for speaker in self._speakers:
                for index in self._wav_index:
                    wav_path = self.wav_path_from_index(speaker, index)
                    feat_path = self.feat_path_from_index(speaker, index)
                    feat = read_binfile(feat_path, self._feature_dim)
                    utt_len = feat.shape[0]
                    speech_dict = {'wav_path': wav_path, 'speaker': speaker, 'utt_index': index,
                                   'feat_path': feat_path, 'utt_len': utt_len}
                    speech_db.append(speech_dict)
            return speech_db
        elif self._split == 'test':
            speech_db = []
            speaker = self._src_speaker
            for index in self._wav_index:
                wav_path = self.wav_path_from_index(speaker, index)
                feat_path = self.feat_path_from_index(speaker, index)
                feat = read_binfile(feat_path, self._feature_dim)
                utt_len = feat.shape[0]
                speech_dict = {'wav_path': wav_path, 'speaker': speaker, 'utt_index': index,
                               'feat_path': feat_path, 'utt_len': utt_len}
                speech_db.append(speech_dict)
            return speech_db
        else:
            print('Unknow split. Use train, val, or test.')
            raise NotImplementedError

    def _get_input_list(self):
        input_list = []
        for i in range(len(self._speech_db)):
            for j in range(self._speech_db[i]['utt_len'] - self._input_len + 1):
                input_list.append([i, j])
        return input_list


    def _speaker_to_one_hot(self, speaker):
        """
        convert speaker to one hot vector
        :param speaker:
        :return:
        """
        speaker_idx = self._speakers.index(speaker)
        one_hot = np.zeros((1, len(self._speakers)), dtype=np.float32)
        one_hot[0, speaker_idx] = 1.0
        return one_hot

    def _remove_energe(self, sample):
        assert (isinstance(sample, np.ndarray))
        return sample[:, 1:]

    def __getitem__(self, index):
        # if self._transform is not None:
        #     seg = self._transform(seg)
        if self._split == 'train' or self._split == 'val':
            utt_idx, seg_idx = self._input_list[index]
            speech_dict = self._speech_db[utt_idx]
            feat = read_binfile(speech_dict['feat_path'], self._feature_dim)
            seg = feat[seg_idx: seg_idx + self._input_len, :]
            orig_seg = self.apply_mvn(seg)
            seg = self._remove_energe(orig_seg)
            label = self._speaker_to_one_hot(speech_dict['speaker'])
            return seg, label
        elif self._split == 'test':
            speech_dict = self._speech_db[index]
            feat = read_binfile(speech_dict['feat_path'], self._feature_dim)
            label = self._speaker_to_one_hot(self._tgt_speaker)
            result_path = self.result_path_at(index)
            orig_feat = self.apply_mvn(feat)
            feat = self._remove_energe(orig_feat)
            return feat, label, orig_feat, result_path
        else:
            raise ValueError('Unknown split, use train, val, or, test instead.')












