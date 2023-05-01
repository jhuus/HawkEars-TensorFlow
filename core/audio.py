# Audio processing, especially extracting and returning spectrograms.

import json
import logging
import random

import cv2
import ffmpeg
import numpy as np
import scipy
import tensorflow as tf

from core import config as cfg
from core import plot

# mel scaling distorts amplitudes, so divide by this to compensate;
# see gen_mel_adjustment.py for the code that generates these values

# this table assumes spec_height = 256, min_audio_freq = 200, max_audio_freq = 10500, etc;
# generate a new table if you change audio parameters
adjust_mel_amplitude = [
    0.026572,0.012701,0.037636,0.012452,0.028451,0.024592,0.015073,0.038828,0.016731,0.022519,0.034229,0.018634,0.025643,0.033347,0.020797,0.024337,
    0.036711,0.022809,0.023758,0.034431,0.029489,0.025575,0.026442,0.036213,0.030292,0.028078,0.028864,0.030484,0.038380,0.030363,0.031084,0.031775,
    0.032447,0.033103,0.038331,0.034675,0.034375,0.034971,0.035542,0.036115,0.036673,0.037223,0.037742,0.038256,0.038759,0.039244,0.039710,0.040178,
    0.040627,0.041059,0.041512,0.041946,0.042342,0.042748,0.043155,0.044366,0.048156,0.044652,0.045008,0.045346,0.045691,0.050217,0.049578,0.046989,
    0.047297,0.052738,0.051687,0.048466,0.052065,0.055743,0.049532,0.055626,0.055306,0.051081,0.062086,0.051290,0.061375,0.054566,0.061572,0.056324,
    0.063672,0.056135,0.067533,0.058231,0.064737,0.064658,0.061921,0.068578,0.066178,0.065350,0.067917,0.071750,0.068495,0.069705,0.070887,0.072174,
    0.075504,0.073687,0.074772,0.075806,0.076842,0.077829,0.078790,0.079726,0.080674,0.081557,0.082439,0.085518,0.085081,0.085372,0.086159,0.086934,
    0.092381,0.089027,0.089538,0.095194,0.092434,0.094044,0.097747,0.094869,0.100948,0.097519,0.102058,0.101943,0.101415,0.107753,0.104564,0.106196,
    0.108815,0.111227,0.110673,0.112332,0.113888,0.115398,0.116886,0.118344,0.119764,0.121134,0.122467,0.123764,0.126699,0.127893,0.127981,0.129752,
    0.134256,0.131860,0.136712,0.135982,0.139942,0.138638,0.144703,0.142790,0.144984,0.148609,0.149659,0.150926,0.152958,0.154968,0.156945,0.158852,
    0.160709,0.162501,0.165109,0.167690,0.168107,0.170537,0.174530,0.173457,0.179523,0.178148,0.182376,0.184516,0.185808,0.188439,0.191050,0.193503,
    0.195990,0.198390,0.200732,0.203017,0.205201,0.207969,0.211868,0.212030,0.217405,0.217360,0.222301,0.224011,0.226514,0.229629,0.232605,0.235546,
    0.238408,0.241169,0.243905,0.247230,0.251042,0.252254,0.257917,0.258923,0.262584,0.267092,0.269215,0.272689,0.276045,0.279378,0.283793,0.286388,
    0.289274,0.294974,0.296375,0.301214,0.304973,0.308250,0.312275,0.316066,0.319883,0.324607,0.327990,0.331685,0.336635,0.340331,0.344367,0.348822,
    0.353199,0.357506,0.361673,0.366349,0.371275,0.375291,0.379910,0.385375,0.389451,0.394228,0.399883,0.404007,0.408862,0.414778,0.419189,0.424577,
    0.429955,0.435198,0.440294,0.446304,0.451389,0.457344,0.462555,0.468535,0.474324,0.479901,0.486489,0.491617,0.498765,0.504081,0.510345,0.516580,
]

class Audio:
    def __init__(self, path_prefix=''):
        self.have_signal = False
        self.signal = None
        self.path_prefix = path_prefix

    # width of spectrogram is determined by input signal length, and height = cfg.spec_height
    def _get_raw_spectrogram(self, signal, low_band=False):
        s = tf.signal.stft(signals=signal, frame_length=cfg.win_length, frame_step=cfg.hop_length, fft_length=2*cfg.win_length, pad_end=True)
        spec = tf.cast(tf.abs(s), tf.float32)

        if low_band:
            min_freq = cfg.low_band_min_audio_freq
            max_freq = cfg.low_band_max_audio_freq
            spec_height = cfg.low_band_spec_height
            mel_scale = cfg.low_band_mel_scale
        else:
            min_freq = cfg.min_audio_freq
            max_freq = cfg.max_audio_freq
            spec_height = cfg.spec_height
            mel_scale = cfg.mel_scale

        # clip frequencies above max_audio_freq
        clip_idx = int(2 * spec.shape[1] * max_freq / cfg.sampling_rate)
        spec = spec[:, :clip_idx]

        if mel_scale:
            # mel spectrogram
            num_spectrogram_bins = int(spec.shape[-1])
            linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                spec_height, num_spectrogram_bins, cfg.sampling_rate, min_freq, cfg.sampling_rate // 2)
            mel = tf.tensordot(spec, linear_to_mel_matrix, 1)
            mel.set_shape(spec.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
            spec = np.transpose(mel)
            if cfg.mel_amplitude_adjustment:
                spec = (spec.T / adjust_mel_amplitude).T
        else:
            # linear frequency scale (used sometimes for plotting spectrograms)
            spec = cv2.resize(spec.numpy(), dsize=(spec_height, spec.shape[0]), interpolation=cv2.INTER_AREA)
            spec = np.transpose(spec)

        return spec

    # version of get_spectrograms that calls _get_raw_spectrogram separately per offset,
    # which is faster when just getting a few spectrograms from a large recording
    def _get_spectrograms_multi_spec(self, signal, offsets, low_band=False):
        last_offset = (len(signal) / cfg.sampling_rate) - cfg.segment_len
        specs = []
        for offset in offsets:
            index = int(offset*cfg.sampling_rate)
            if offset <= last_offset:
                segment = signal[index:index + cfg.segment_len * cfg.sampling_rate]
            else:
                segment = signal[index:]
                pad_amount = cfg.segment_len * cfg.sampling_rate - segment.shape[0]
                segment = np.pad(segment, ((0, pad_amount)), 'constant', constant_values=0)

            spec = self._get_raw_spectrogram(segment, low_band=low_band)
            specs.append(spec)


        return specs

    # normalize values between 0 and 1
    def _normalize(self, specs):
        for i in range(len(specs)):
            max = specs[i].max()
            if max > 0:
                specs[i] = specs[i] / max

            specs[i] = specs[i].clip(0, 1)

    # stereo recordings sometimes have one clean channel and one noisy one;
    # so rather than just merge them, use heuristics to pick the cleaner one
    def _choose_channel(self, left_channel, right_channel, scale):
        left_signal = scale * np.frombuffer(left_channel, '<i2').astype(np.float32)
        right_signal = scale * np.frombuffer(right_channel, '<i2').astype(np.float32)
        recording_seconds = int(len(left_signal) / cfg.sampling_rate)
        check_seconds = min(recording_seconds, cfg.check_seconds)
        offsets = [0]
        self.signal = left_signal
        left_spec = self.get_spectrograms(offsets, segment_len=check_seconds)[0]
        self.signal = right_signal
        right_spec = self.get_spectrograms(offsets, segment_len=check_seconds)[0]

        if left_spec.sum() > right_spec.sum():
            # more noise in the left channel
            return right_signal, right_channel
        else:
            # more noise in the right channel
            return left_signal, left_channel

    # return a white noise spectrogram with values in range [0, 1]
    def white_noise(self, mean=0, stdev=1):
        samples = cfg.segment_len * cfg.sampling_rate
        segment = np.random.normal(loc=mean, scale=.0001, size=samples).astype(np.float32)
        spec = self._get_raw_spectrogram(segment)
        spec = spec / spec.max() # normalize to [0, 1]
        return spec.reshape((cfg.spec_height, cfg.spec_width, 1))

    # return a spectrogram with a sin wave of the given frequency
    def sin_wave(self, frequency):
        samples = cfg.segment_len * cfg.sampling_rate
        t = np.linspace(0, 2*np.pi, samples)
        segment = np.sin(t*frequency*cfg.segment_len)
        spec = self._get_raw_spectrogram(segment)
        spec = spec / spec.max() # normalize to [0, 1]
        return spec.reshape((cfg.spec_height, cfg.spec_width, 1))

    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this;
    # if raw_spectrograms array is specified, populate it with spectrograms before normalization
    def get_spectrograms(self, offsets, segment_len=cfg.segment_len, spec_exponent=cfg.spec_exponent, low_band=False, multi_spec=False, raw_spectrograms=None):
        if not self.have_signal:
            return None

        if multi_spec:
            # call _get_raw_spectrogram separately per offset, which is faster when just getting a few spectrograms from a large recording
            specs = self._get_spectrograms_multi_spec(self.signal, offsets)
        else:
            # call _get_raw_spectrogram for the whole signal, then break it up into spectrograms;
            # this is faster when getting overlapping spectrograms for a whole recording
            spectrogram = None
            spec_width_per_sec = int(cfg.spec_width / cfg.segment_len)
            # create in blocks so we don't run out of GPU memory
            block_length = cfg.spec_block_seconds * cfg.sampling_rate
            start = 0
            i = 0
            while start < len(self.signal):
                i += 1
                length = min(block_length, len(self.signal) - start)
                block = self._get_raw_spectrogram(self.signal[start:start+length], low_band=low_band)
                if spectrogram is None:
                    spectrogram = block
                else:
                    spectrogram = np.concatenate((spectrogram, block), axis=1)

                start += length

            last_offset = (spectrogram.shape[1] / spec_width_per_sec) - cfg.segment_len

            specs = []
            for offset in offsets:
                if offset <= last_offset:
                    specs.append(spectrogram[:, int(offset * spec_width_per_sec) : int((offset + segment_len) * spec_width_per_sec)])
                else:
                    spec = spectrogram[:, int(offset * spec_width_per_sec):]
                    spec = np.pad(spec, ((0, 0), (0, cfg.spec_width - spec.shape[1])), 'constant', constant_values=0)
                    specs.append(spec)

        if raw_spectrograms is not None and len(raw_spectrograms) == len(specs):
            for i, spec in enumerate(specs):
                raw_spectrograms[i] = spec

        self._normalize(specs)
        if spec_exponent != 1:
            for i in range(len(specs)):
                specs[i] = specs[i] ** spec_exponent

        return specs

    def signal_len(self):
        return len(self.signal) if self.have_signal else 0

    def load(self, path, keep_bytes=False):
        self.have_signal = False
        self.signal = None
        spectrogram = None

        try:
            self.have_signal = True
            scale = 1.0 / float(1 << ((16) - 1))
            info = ffmpeg.probe(path)

            if not 'channels' in info['streams'][0].keys() or info['streams'][0]['channels'] == 1:
                bytes, _ = (ffmpeg
                    .input(path)
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.sampling_rate}')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True))

                # convert byte array to float array, and then to a numpy array
                self.signal = scale * np.frombuffer(bytes, '<i2').astype(np.float32)
            else:
                left_channel, _ = (ffmpeg
                    .input(path)
                    .filter('channelsplit', channel_layout='stereo', channels='FL')
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.sampling_rate}')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True))

                right_channel, _ = (ffmpeg
                    .input(path)
                    .filter('channelsplit', channel_layout='stereo', channels='FR')
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.sampling_rate}')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True))

                self.signal, bytes = self._choose_channel(left_channel, right_channel, scale)

            if keep_bytes:
                self.bytes = bytes # when we want the raw audio, e.g. to write a segment to a wav file

        except ffmpeg.Error as e:
            self.have_signal = False
            tokens = e.stderr.decode().split('\n')
            if len(tokens) >= 2:
                logging.error(f'Caught exception in audio load: {tokens[-2]}')
            else:
                logging.error(f'Caught exception in audio load')

        logging.debug('Done loading audio file')
        return self.signal, cfg.sampling_rate
