import pyworld as pw
from config.config import mfcc_config, mcep_config

from src.dataset.feature_extraction.mcep import spec2mcep
from src.dataset.feature_extraction.mfcc import spec2mfcc
import numpy as np


def speech_analysis(wav, fs, f0_lower=50.0, f0_upper=400.0, shift=5.0):
    f0, t = pw.harvest(np.double(wav), fs, f0_lower, f0_upper, shift)
    f0 = pw.stonemask(wav, f0, t, fs)
    vuv = f0 != 0
    # make the data type to be float32 for the convenience of the calculation in pyTorch
    spectrogram = np.float32(pw.cheaptrick(wav, f0, t, fs))
    ap = pw.d4c(wav, f0, t, fs)
    mfcc = spec2mfcc(spectrogram, fs, mfcc_config['dim'])
    mcep = spec2mcep(spectrogram, mcep_config['dim'])
    utt = {'fs': fs, 'shift': shift,
           'f0': f0, 'vuv': vuv,
           'spectrogram': spectrogram, 'ap': ap,
           'mfcc': mfcc, 'mcep': mcep}
    return utt

def speech_synthesis(utt):
    utt['spectrogram'] = np.double(utt['spectrogram'])
    wav_h = pw.synthesize(utt['f0'], utt['spectrogram'], utt['ap'], utt['fs'], utt['shift'])
    return wav_h





