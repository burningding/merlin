import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from . import *
from .dataset_utils import *
import sys


class SpeechDB(Dataset):
    """
    base dataset for speech data
    """
    def __init__(self, split, dataset_path, exp_path, use_mvn=False, feature_type=None, feature_dim=None):
        self._split = split
        self._dataset_path = dataset_path
        self._exp_path = exp_path
        self._feature_type = feature_type if feature_type else 'mgc'
        self._feature_dim = feature_dim if feature_dim else 60
        self._wav_index = []
        self._speech_db = []
        self._mvn_params_path = ''
        self._use_mvn = use_mvn
        self._mvn_params = None

    def speaker_feat_path(self, speaker):
        raise NotImplementedError

    def feat_path_from_index(self, speaker, index, sep='a'):
        raise NotImplementedError

    def feat_path_at(self, i):
        raise NotImplementedError

    def speaker_pitch_path(self, speaker):
        raise NotImplementedError

    def pitch_path_from_index(self, speaker, index, sep='a'):
        raise NotImplementedError

    def pitch_path_at(self, i):
        raise NotImplementedError

    def speaker_result_path(self):
        raise NotImplementedError

    def result_path_from_index(self, index, sep='a'):
        raise NotImplementedError

    def result_path_at(self, i):
        raise NotImplementedError

    def _get_default_feature_path(self):
        raise NotImplementedError

    def _get_default_result_path(self):
        raise NotImplementedError

    def _get_default_mvn_params_path(self):
        raise NotImplementedError

    def _get_index_from_split(self):
        raise NotImplementedError

    def _get_input_len(self):
        raise NotImplementedError

    def _generate_speech_db(self):
        raise NotImplementedError

    def _get_input_list(self):
        raise NotImplementedError

    def build_pitch_model(self, speaker):
        """
        Build pitch model for give speaker
        :param speaker:
        :return: pitch model:
                        - data: f0
                        - logmean
                        - logstd
                        - speaker
                        - utts: utterance indices
        """
        f0 = []
        for index in self._wav_index:
            utt_fname = self.feat_path_from_index(speaker, index)
            lf0 = self.read_binfile(utt_fname, dim=1)
            utt = torch.load(utt_fname)
            f0.append(utt['f0'][utt['vuv']])
        f0 = np.hstack(f0)
        model = {'data': f0}
        logdata = np.log(f0[f0 > 0])
        model['logmean'] = np.mean(logdata)
        model['logstd'] = np.std(logdata)
        model['speaker'] = speaker
        model['utts'] = self._wav_index
        self.save_pitch_model(model, speaker)
        return model

    def save_pitch_model(self, model, speaker):
        pitch_path = self.speaker_pitch_path(speaker)
        torch.save(model, pitch_path)

    def load_pitch_model(self, speaker):
        pitch_path = self.speaker_pitch_path(speaker)
        if not os.path.exists(pitch_path):
            return None
        return torch.load(pitch_path)

    def compute_mvn(self, path):
        n = 0.
        x = 0.
        x2 = 0.
        min_x = float(sys.maxsize)
        n_utts = 0
        for speech_dict in self._speech_db:
            utt_feats = read_binfile(speech_dict['feat_path'], self._feature_dim)
            x += np.sum(utt_feats, axis=0, keepdims=True)
            x2 += np.sum(utt_feats ** 2, axis=0, keepdims=True)
            min_utt = np.amin(utt_feats)
            min_x = min(min_x, min_utt)
            n += utt_feats.shape[0]
            n_utts += 1
            if n_utts % 100 == 0:
                info("accumulated %s utts" % n_utts)
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        info("mean shape is %s, value is\n%s" % (mean.shape, mean))
        info("std shape is %s, value is\n%s" % (std.shape, std))
        mvn_params = {"mean": mean, "std": std}
        check_and_makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            info("dumping mvn params to %s" % path)
            pickle.dump(mvn_params, f)
        return mvn_params

    def apply_mvn(self, sample):
        assert (isinstance(sample, np.ndarray))
        if self._mvn_params is None:
            return sample
        else:
            mean = self._mvn_params["mean"]
            std = self._mvn_params["std"]
            return (sample - mean) / std

    def undo_mvn(self, sample):
        assert (isinstance(sample, np.ndarray))
        if self._mvn_params is None:
            return sample
        else:
            mean = self._mvn_params["mean"]
            std = self._mvn_params["std"]
            return sample * std + mean

    def _get_mvn_params(self):
        if not self._use_mvn:
            return None
        if self._split == 'train':
            info("In training, comuting mvn parameters")
            mvn_params = self.compute_mvn(self._mvn_params_path)
        elif self._split == 'test' or self._split == 'val':
            info("In testing or validation, loading mvn paramters from{}".format(self._mvn_params_path))
            with open(self._mvn_params_path, 'rb') as f:
                mvn_params = pickle.load(f)
        else:
            raise ValueError("Unknown split, use train, val, or test")
        return mvn_params

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError











