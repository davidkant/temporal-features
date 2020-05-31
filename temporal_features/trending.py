import librosa
from sklearn.decomposition import PCA

import numpy as np
import math


def trending(
    y=None,
    y_pca=None,
    win=60,
    feature='STFT',
    n_fft=2048,
    hop_length=1024,
    sr=44100,
    verbose=True,
):
    """FFT trend analysis.

    Analyze a signal for long-term change, or trends, using FFT analysis. The audio
    signal is first transformed to the frequency domain and then reduced to one
    dimension using PCA. FFT trend analysis is performed on the dimenion-reduced
    frequency-domain signal.

      :: y -> freq-domain -> PCA -> FFT trend analysis

    Note: win length is a request and will be quantized to nearest hop.

    Note: signal will be cropped to size and/or zero padded to fit analysis window.

    Args:
        y (np.ndarray [shape=(n,)]): The input signal to be analyzed.
        y_pca (np.ndarray [shape=(n,)], optional): Alt input for a PCA'd signal.
        win (number > 0 [scalar]): Analysis window legth in seconds.
        feature ({'STFT', 'CQT', 'Mel'}): Frequency-domain transform.
        n_fft (int): Size of the FFT.
        hop_length (int): Hop length for frequency-domain transform.
        sr (int): Audio sample rate.
        verbose (bool): Tell me about it.

    Returns:
        alpha (np.ndarray [shape=(n,)]): FFT spectrum, complex.
        bin_freqs (np.ndarray [shape=(n,)]): FFT bin frequencies.
    """

    if y_pca is None:

        # frequency domain transform
        Y = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

        # PCA to one dim
        pca = PCA(n_components=1)
        Y_pca = np.squeeze(pca.fit_transform(Y.T))

    else:
        Y_pca = y_pca


    # format window, crop if too large, zero pad if too small
    win_in_samps = int(math.ceil(win * sr / float(hop_length)))

    if len(Y_pca) >= win_in_samps:
        Y_pca_window = Y_pca[:win_in_samps]

    elif len(Y_pca) < win_in_samps:
        Y_pca_window = np.pad(Y_pca, (0, win_in_samps - len(Y_pca)), 'constant')

    # FFT analysis
    ALPHA = np.fft.rfft(Y_pca_window)

    # compute fundamental frequency
    ffreq = ((len(ALPHA) * 2 * hop_length) / float(sr))

    # compute bin frequencies
    bin_freqs = ffreq / 60.0 / np.arange(0, len(ALPHA))

    # sanity check
    if verbose:
      print("SANITY CHECK!")
      print("- input shape: {}".format(y.shape))
      print("- STFT shape: {}".format(Y.shape))
      print("- PCA shape: {}".format(Y_pca.shape))
      print("- padded shape: {}".format(Y_pca_padded.shape))
      print("- zero padding: {}".format(len(Y_pca_padded) - len(Y_pca)))
      print("- analysis bins: {}".format(len(ALPHA) - 1))
      print("- analysis freq: {}".format(ffreq))

    return ALPHA, bin_freqs


def tr60(y=None, pca=None, verbose=False):
    """Trending energy at 60 seconds and below."""
    ALPHA, bin_freqs = trending(y=y, pca=pca, win=60, verbose=verbose)
    return np.abs(ALPHA)[1]


def tr600(y=None, pca=None, verbose=False):
    """Trending energy at 10 minutes and below."""
    ALPHA, bin_freqs = trending(y=y, pca=pca, win=600, verbose=verbose)
    return np.abs(ALPHA)[1]

