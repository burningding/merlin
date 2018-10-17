import torch
from sklearn.preprocessing import normalize
from .dataset_utils import *
from ..utils.logger import *


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        transforms.Compose([
        transforms.Scale(),
        transforms.PadTrim(max_len=16000),
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class LogTransform(object):
    """
    Transform the spectrogram to log spectrogram
    """
    def __init__(self):
        pass

    def __call__(self, fea):
        """
        :param fea: speech feature (often spectrogram)
        :return: log scale speech feature
        """
        return torch.log(fea)


class Normalization(object):
    """
    Normalize the speech feature to have unit norm
    """
    def __init__(self, is_training):
        self._is_training = is_training

    def __call__(self, fea):
        """
        :param fea: speech feature (often spectrogram)
        :return: speech feature with unit norm and the computed norm
        """
        fea, norm = normalize(fea, return_norm=True)
        if self._is_training:
            return fea
        else:
            return fea, norm


