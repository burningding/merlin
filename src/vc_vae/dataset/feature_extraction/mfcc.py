import numpy as np
from scipy.fftpack import dct, idct
from scipy.signal import convolve2d


def hz2mel(f, htk = False):
    if htk:
        z = np.multiply(2595, np.log10(np.add(1, np.divide(f, 700))))
    else:
        f_0 = 0.0
        f_sp = 200 / 3
        brkfrq = 1000
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27.0)

        f = np.array(f, ndmin = 1)
        z = np.zeros((f.shape[0], ))

        for i in range(f.shape[0]):
            if f[i] < brkpt:
                z[i] = (f[i] - f_0) / f_sp
            else:
                z[i] = brkpt + (np.log(f[i] / brkfrq) / np.log(logstep))
    return z

def mel2hz(z, htk = False):
    if htk:
        f = np.multiply(700, np.subtract(np.power(10, np.divide(z, 2595)), 1))
    else:
        f_0 = 0
        f_sp = 200/ 3
        brkfrq = 1000
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27.0)

        z = np.array(z, ndmin=1)
        f = np.zeros((z.shape[0],))

        for i in range(z.shape[0]):
            if z[i] < brkpt:
                f[i] = f_0 + f_sp * z[i]
            else:
                f[i] = brkfrq * np.exp(np.log(logstep) * (z[i] - brkpt))
    return f


def fft2melmx(fft_length, fs, nfilts=0, band_width=1, min_freq=0, max_freq=0,
              htk=False, constamp=False):
    if nfilts == 0:
        nfilts = np.ceil(hz2mel(max_freq, htk) / 2)
    if max_freq == 0:
        max_freq = fs / 2

    wts = np.zeros((int(nfilts), int(fft_length)))
    fftfrqs = np.multiply(np.divide(np.arange(0.0, fft_length / 2.0 + 1.0), fft_length), fs)

    min_mel = hz2mel(min_freq, htk)
    max_mel = hz2mel(max_freq, htk)
    binfrqs = mel2hz(np.add(min_mel, np.multiply(np.arange(0.0, nfilts + 2.0),
                                                 (max_mel - min_mel) / (nfilts + 1.0))), htk)

    for i in range(int(nfilts)):
        fs_tmp = binfrqs[np.add(np.arange(0, 3), i)]
        fs_tmp = np.add(fs_tmp[1], np.multiply(band_width, np.subtract(fs_tmp, fs_tmp[1])))
        loslope = np.divide(np.subtract(fftfrqs, fs_tmp[0]), np.subtract(fs_tmp[1], fs_tmp[0]))
        hislope = np.divide(np.subtract(fs_tmp[2], fftfrqs), np.subtract(fs_tmp[2], fs_tmp[1]))
        wts[i, 0: int(fft_length / 2) + 1] = np.maximum(0, np.minimum(loslope, hislope))

    if constamp == False:
        wts = np.matmul(np.diag(np.divide(2, np.subtract(binfrqs[2: int(nfilts) + 2],
                                                         binfrqs[0: int(nfilts)]))), wts)

    return wts


def spec2mfcc(spec, fs, nmel=13, delta=0):
    """
    Convert spectrogram into MFCCs using DCT
    :param spec: spectral envelope
    :param fs: sampling rate of acoustic waveform
    :param nmel: number of mel cepstral components to compute
    :param delta: number of delta values to computer [0-2]
    :return: mfcc - [N x nmel*(delta+1)] matrix of cepstral values on derivatives
    """
    # Generate filter bank
    fbank = fft2melmx(spec.shape[1] * 2 - 2, fs, nmel, 1, 0, fs / 2, 1)
    fbank = fbank[:, :spec.shape[1]]
    fbank = np.transpose(fbank)

    # Mel -> log -> DCT
    mfcc = dct(np.matmul(np.log(spec), fbank), norm='ortho')

    # Smoothing kernel for delta and delta-delta
    d1ker = np.array([-1, 8, 0, -8, 1]) / 12
    d2ker = np.array([-1, 16, -30, 16, -1]) / 12

    # Include delta and delta-delta coefficients
    if delta == 1:
        d1 = convolve2d(mfcc, d1ker, 'same')
        mfcc = np.concatenate((mfcc, d1), axis=0)
    elif delta == 2:
        d1 = convolve2d(mfcc, d1ker, 'same')
        d2 = convolve2d(mfcc, d2ker, 'same')
        mfcc = np.concatenate((mfcc, d1, d2), axis=0)

    return mfcc

def mfcc2spec(mfcc, fs, nSp):
    """
    Convert MFCCs into spectrogram
    :param mfcc: MFCC caluclated from spectrogram using mfcc = dct(fbank*(log(s)))
    :param fs: nyquist rate for interested badwidth. (16000 if we want the filter bank to be created from 0  to 8000 Hz)
    :param nSp: number of spectral points we want to extrapolate to from 0 to fs/2
    :return: spectrogram
    """
    nmel = mfcc.shape[1]

    # Generate filter bank
    fbank = fft2melmx(nSp * 2 - 2, fs, nmel, 1, 0, fs / 2, 1)
    fbank = fbank[:, : nSp]
    fbank = np.transpose(fbank)
    fbank_conv_log_spec = idct(mfcc, norm='ortho')
    log_spec = np.matmul(fbank_conv_log_spec, np.linalg.pinv(fbank))

    spec = np.exp(log_spec)
    r = np.argmax(fbank[:, -1])
    spec[:, r:] = np.finfo('float64').eps
    return spec
