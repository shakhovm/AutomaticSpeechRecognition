import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class MFCC:
    def __init__(self, FFT_size, mel_filter_num, dct_filter_num, window=None):
        self.FFT_size = FFT_size
        self.mel_filter_num = mel_filter_num
        if window is None:
            window = signal.windows.get_window("hann", FFT_size, fftbins=True)
        self.window = window
        self.dct_filter_num = dct_filter_num

    @staticmethod
    def freq_to_mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    @staticmethod
    def mel_to_freq(mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    @staticmethod
    def normalize_audio(audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    def get_filter_points(self, fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = self.freq_to_mel(fmin)
        fmax_mel = self.freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
        freqs = self.mel_to_freq(mels)

        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    @staticmethod
    def get_filters(filter_points, FFT_size):
        filters = np.zeros((len(filter_points) - 2, int(FFT_size / 2 + 1)))

        for n in range(len(filter_points) - 2):
            filters[n, filter_points[n]: filter_points[n + 1]] = np.linspace(0, 1,
                                                                             filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1]: filter_points[n + 2]] = np.linspace(1, 0,
                                                                                 filter_points[n + 2] - filter_points[
                                                                                     n + 1])

        return filters

    @staticmethod
    def dct(dct_filter_num, filter_len):
        basis = np.empty((dct_filter_num, filter_len))
        basis[0, :] = 1.0 / np.sqrt(filter_len)

        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

        for i in range(1, dct_filter_num):
            basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        return basis

    def get_dct(self, audio, sample_rate, noverlap=904):
        normilized_audio = self.normalize_audio(audio)
        normilized_audio[1:] -= 0.5*normilized_audio[:-1]
        padded_audio = np.pad(normilized_audio, int(self.FFT_size / 2), mode='reflect')
        f, t, Sxx = signal.spectrogram(padded_audio, sample_rate, nperseg=self.FFT_size,
                                       noverlap=noverlap, window=self.window)
        self.Sxx = Sxx
        self.f = f
        self.t = t
        self.freq_min = 0
        self.freq_high = sample_rate / 2
        self.filter_points, self.mel_freqs = self.get_filter_points(self.freq_min,
                                                          self.freq_high, self.mel_filter_num, self.FFT_size,
                                                          sample_rate=sample_rate)
        self.filters = self.get_filters(self.filter_points, self.FFT_size)
        enorm = 2.0 / (self.mel_freqs[2:self.mel_filter_num + 2] - self.mel_freqs[:self.mel_filter_num])
        self.filters *= enorm[:, np.newaxis]

        self.audio_filtered = np.dot(self.filters, Sxx)


        dct_filters = self.dct(self.dct_filter_num, self.mel_filter_num)

        cc = np.dot(dct_filters, 10 * np.log10(self.audio_filtered))
        return cc

    def visualize_Sxx(self):
        plt.pcolormesh(self.t, self.f, np.log(self.Sxx), shading='gouraud')
        plt.colorbar()
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def visualize_mel(self):
        plt.pcolormesh(np.arange(self.audio_filtered.shape[1]), np.arange(self.audio_filtered.shape[0]),
                       np.log(self.audio_filtered), shading='gouraud')
        plt.colorbar()
        plt.show()
        # for n in range(self.filters.shape[0]):
        #     plt.plot(self.filters[n])
