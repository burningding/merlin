import numpy as np


def build_pitch_model(dataset, speaker_name):
    """
    Build the pitch model for pitch conversion
    :param dataset: An speechDB dataset instance
    :param speaker_name: The speaker name to be build
    :return: pitch model
    """
    pitch_model = dataset.load_pitch_model(speaker_name)
    if pitch_model is None:
        pitch_model = dataset.build_pitch_model(speaker_name)
    return pitch_model


def pitch_conversion(f0, src_pitch_model, tgt_pitch_model):
    if src_pitch_model['speaker'] == tgt_pitch_model['speaker']:
        return f0

    def log_transform(f0, src_pitch_model, tgt_pitch_model):
        converted_f0 = np.exp((np.log(f0 + np.finfo(np.float32).eps) - src_pitch_model['logmean']) * \
                       (tgt_pitch_model['logstd'] / src_pitch_model['logstd']) + tgt_pitch_model['logmean'])
        return converted_f0
    conv_f0 = log_transform(f0, src_pitch_model, tgt_pitch_model)
    conv_f0[conv_f0 < 10] = 0
    return conv_f0