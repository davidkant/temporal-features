import temporal_features.trending as tr

import librosa
import numpy as np


def test_trending_dunn_win_600():
    filename = 'resources/Thresholds_and_Fragile_States_10_19_2012.mp3'
    y, sr = librosa.load(filename, mono=True, sr=None)
    alpha, bins = tr.trending(y, win=600, pca_random_state=0, verbose=False)
    alpha_test = np.load('test/data/trending_dunn_win_600_alpha_200608.npy')
    bins_test = np.load('test/data/trending_dunn_win_600_bins_200608.npy')
    np.testing.assert_array_almost_equal(alpha, alpha_test)
    np.testing.assert_array_almost_equal(bins, bins_test)
    print('OK')
    return 1

def test_target_spec_dunn_win_600():
    alpha = np.load('test/data/trending_dunn_win_600_alpha_200608.npy')
    bins = np.load('test/data/trending_dunn_win_600_bins_200608.npy')
    target_spec = [
        np.inf,
        600,
        300,
        200,
        150,
        120,
        100,
        85,
        70,
        60,
        50,
        40,
        30,
        20,
        15,
        10,
        5,
        4,
        3,
        2,
        1,
        0,
    ]
    new_spec = tr.FFT_to_target_spec(alpha, bins, target_spec, aggr_func=np.max)
    new_spec_test = np.load('test/data/target_spec_dunn_win_600_200608.npy')
    np.testing.assert_array_almost_equal(new_spec, new_spec_test)
    print('OK')
    return 1

def test_temporal_centroid():
    alpha = np.load('test/data/trending_dunn_win_600_alpha_200608.npy')
    bins = np.load('test/data/trending_dunn_win_600_bins_200608.npy')
    centroid = tr.temporal_centroid(alpha, bins)
    assert centroid == 7.847722156318984
    print('OK')
    return 1

def test_temporal_centroid_custom_spec():
    alpha = np.load('test/data/trending_dunn_win_600_alpha_200608.npy')
    bins = np.load('test/data/trending_dunn_win_600_bins_200608.npy')
    target_spec = [
        np.inf,
        600,
        300,
        200,
        150,
        120,
        100,
        85,
        70,
        60,
        50,
        40,
        30,
        20,
        15,
        10,
        5,
        4,
        3,
        2,
        1,
        0,
    ]
    new_spec = tr.FFT_to_target_spec(alpha, bins, target_spec, aggr_func=np.max)
    new_spec_test = np.load('test/data/target_spec_dunn_win_600_200608.npy')
    centroid = tr.temporal_centroid(new_spec, target_spec)
    assert centroid == 113.70549064467077
    print('OK')
    return 1

# run all tests
test_trending_dunn_win_600()
test_target_spec_dunn_win_600()
test_temporal_centroid()
test_temporal_centroid_custom_spec()
