import numpy as np
import os
from ..dataset.dataset_utils import *
import pickle
from .model_io import *
from logger import *


def build_pitch_model(speakers, args):
    """
    build pitch models for voice conversion
    :param speakers: list, the speakers whose pitch models are to be built
    :param args: namespace, the args parameter of the main function
    :return: None, all the pitch models are saved
    """
    num_train_utt = args.num_train_utt
    dataset_path = args.dataset_path
    exp_dir = args.exp_dir
    name_format = args.name_format
    utt_index = [i for i in range(1, 1 + num_train_utt)]
    for speaker in speakers:
        lf0_path = os.path.join(dataset_path, 'feature', speaker, 'lf0')
        lf0s = []
        for idx in utt_index:
            lf0 = read_binfile(os.path.join(lf0_path, name_format.format('a', idx) + '.lf0'), dim=1)
            lf0s.append(lf0[lf0 > 0])
        lf0s = np.hstack(lf0s)
        model = {'logmean': np.mean(lf0s), 'logstd': np.std(lf0s)}
        save_pitch_model(model, os.path.join(exp_dir, 'pitch_model', speaker + '.pkl'))
        info('saved the pitch model of {0} to {1}'.format(speaker, os.path.join(exp_dir, 'pitch_model', speaker + '.pkl')))


def pitch_conversion(lf0, src_pitch_model, tgt_pitch_model):
    """
    convert pitch for the test utterance
    :param lf0: ndarray, log-f0
    :param src_pitch_model: dict, pitch model of the source speaker
    :param tgt_pitch_model: dict, pitch model of the target speaker
    :return: ndarray, converted log-f0
    """
    def log_transform(lf0, src_pitch_model, tgt_pitch_model):
        converted_lf0 = (lf0 - src_pitch_model['logmean']) * \
                              (tgt_pitch_model['logstd'] / src_pitch_model['logstd']) + tgt_pitch_model['logmean']
        return converted_lf0

    conv_lf0 = log_transform(lf0, src_pitch_model, tgt_pitch_model)
    return conv_lf0



# def pitch_conversion(f0, src_pitch_model, tgt_pitch_model):
#     if src_pitch_model['speaker'] == tgt_pitch_model['speaker']:
#         return f0
#
#     def log_transform(f0, src_pitch_model, tgt_pitch_model):
#         converted_f0 = np.exp((np.log(f0 + np.finfo(np.float32).eps) - src_pitch_model['logmean']) * \
#                        (tgt_pitch_model['logstd'] / src_pitch_model['logstd']) + tgt_pitch_model['logmean'])
#         return converted_f0
#     conv_f0 = log_transform(f0, src_pitch_model, tgt_pitch_model)
#     conv_f0[conv_f0 < 10] = 0
#     return conv_f0