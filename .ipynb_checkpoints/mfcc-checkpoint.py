import numpy as np

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = met_to_freq(mels)

    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs


def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points) - 2, int(FFT_size / 2 + 1)))

    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[
            n + 1])

    return filters


def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)

    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

    return basis