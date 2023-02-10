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

# this table is for spec_height = 128
adjust_mel_amplitude_128 = [
    0.027535,0.027970,0.028387,0.033496,0.029787,0.029907,0.034109,0.033365,0.031747,0.038496,0.032048,0.040483,0.034425,0.040754,0.038629,0.038985,
    0.043749,0.041999,0.042933,0.044365,0.045721,0.047000,0.048250,0.049407,0.050525,0.051583,0.052599,0.053557,0.056935,0.056725,0.057220,0.061665,
    0.059748,0.063860,0.064301,0.065353,0.067451,0.070080,0.070942,0.072556,0.074714,0.076987,0.078040,0.079532,0.083154,0.083639,0.087098,0.087802,
    0.090903,0.093527,0.095046,0.097454,0.100190,0.103046,0.104504,0.107881,0.110178,0.113631,0.115582,0.118764,0.121783,0.124639,0.127617,0.131809,
    0.133528,0.137826,0.141227,0.144311,0.147876,0.152447,0.154864,0.159781,0.163070,0.167436,0.171540,0.175474,0.180503,0.184199,0.189320,0.194113,
    0.198737,0.203105,0.209038,0.213551,0.219032,0.224439,0.230364,0.235701,0.241589,0.247416,0.254256,0.259819,0.266439,0.272926,0.280244,0.286657,
    0.293751,0.301616,0.308465,0.316261,0.324171,0.332089,0.340515,0.348590,0.357682,0.366413,0.375351,0.384754,0.394475,0.403824,0.414084,0.424323,
    0.434811,0.445567,0.456793,0.467964,0.479627,0.491572,0.503620,0.515996,0.529156,0.542049,0.555417,0.569471,0.583207,0.597816,0.612573,0.627649,
]

# this table is for spec_height = 256
adjust_mel_amplitude_256 = [
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
    def _get_raw_spectrogram(self, signal):
        s = tf.signal.stft(signals=signal, frame_length=cfg.win_length, frame_step=cfg.hop_length, fft_length=2*cfg.win_length, pad_end=True)

        spec = tf.cast(tf.abs(s), tf.float32)

        # clip frequencies above max_audio_freq
        num_freqs = spec.shape[1]
        clip_idx = int(2 * spec.shape[1] * cfg.max_audio_freq / cfg.sampling_rate)
        spec = spec[:, :clip_idx]

        if cfg.mel_scale:
            # mel spectrogram
            num_spectrogram_bins = int(spec.shape[-1])
            linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                cfg.spec_height, num_spectrogram_bins, cfg.sampling_rate, cfg.min_audio_freq, cfg.sampling_rate // 2)
            mel = tf.tensordot(spec, linear_to_mel_matrix, 1)
            mel.set_shape(spec.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
            spec = np.transpose(mel)
            if cfg.mel_amplitude_adjustment:
                if cfg.spec_height == 128:
                    spec = (spec.T / adjust_mel_amplitude_128).T
                else:
                    # assume spec_height = 256
                    spec = (spec.T / adjust_mel_amplitude_256).T
        else:
            # linear frequency scale (used sometimes for plotting spectrograms)
            spec = cv2.resize(spec.numpy(), dsize=(cfg.spec_height, spec.shape[0]), interpolation=cv2.INTER_AREA)
            spec = np.transpose(spec)

        return spec

    # version of get_spectrograms that calls _get_raw_spectrogram separately per offset,
    # which is faster when just getting a few spectrograms from a large recording
    def _get_spectrograms_multi_spec(self, signal, offsets):
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

            spec = self._get_raw_spectrogram(segment)
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
        seconds = int(len(left_signal) / cfg.sampling_rate)
        max_offset = min(seconds - cfg.segment_len, 3 * cfg.segment_len) # look at the first 3 non-overlapping segments
        offsets = [i for i in range(0, max_offset, cfg.segment_len)]
        self.signal = left_signal
        left_specs = self.get_spectrograms(offsets, multi_spec=True)
        self.signal = right_signal
        right_specs = self.get_spectrograms(offsets, multi_spec=True)
        left_sum = 0
        for spec in left_specs:
            left_sum += spec.sum()

        right_sum = 0
        for spec in right_specs:
            right_sum += spec.sum()

        if left_sum > right_sum:
            # more noise in the left channel
            return right_signal, right_channel
        else:
            # more noise in the right channel
            return left_signal, left_channel

    # look for a sound in the given frequency (height) range, in the first segment of the spec;
    # if found, return the starting offset in seconds
    def _find_sound(self, spec, sound_factor, min_height, max_height):
        # get sum of magnitude per sample in the given frequency range
        spec = spec.transpose()
        sum = np.zeros(spec.shape[0])
        for sample_offset in range(spec.shape[0]):
            sum[sample_offset] = np.sum(spec[sample_offset][min_height:max_height])

        # find samples more than sound_factor times as loud as median
        first_sound_sample = 0
        found = False
        median = np.median(sum)
        sound = np.zeros(spec.shape[0])
        for sample_offset in range(spec.shape[0]):
            if sum[sample_offset] > sound_factor * median:
                sound[sample_offset] = 1
                if not found:
                    first_sound_sample = sample_offset
                    found = True

        if found and first_sound_sample < cfg.spec_width:
            # we found sound in the first segment, so try to center it;
            # scan backwards from end of potential segment until non-silence
            curr_offset = first_sound_sample
            end_offset = min(curr_offset + cfg.spec_width - 1, spec.shape[0] - 1)

            while sound[end_offset] == 0:
                end_offset -= 1

            # determine padding and back up from curr_offset to center the sound
            sound_len = end_offset - curr_offset + 1
            total_pad_len = cfg.spec_width - sound_len
            initial_pad_len = min(int(total_pad_len / 2), curr_offset)
            start_offset = curr_offset - initial_pad_len
            return True, start_offset

        else:
            return False, 0

    # find sounds in audio, using a simple heuristic approach;
    # return array of floats representing offsets in seconds where significant sounds begin;
    # when analyzing an audio file, the only goal is to minimize splitting of bird sounds,
    # since sound fragments are often misidentified;
    # when extracting data for training we also want to discard intervals that don't seem
    # to have significant sounds in them, but in analysis it's better to let the neural net
    # decide if a spectrogram contains a bird sound or not
    def find_sounds(self, signal=None, sound_factor=1.15, keep_empty=True):
        if signal is None:
            if self.have_signal:
                signal = self.signal
            else:
                return []

        num_seconds = signal.shape[0] / cfg.sampling_rate
        samples_per_sec = cfg.spec_width / cfg.segment_len
        freq_cutoff = int(cfg.spec_height / 4) # dividing line between "low" and "high" frequencies

        # process in chunks, where each is as wide as two spectrograms
        curr_start = 0
        offsets = []
        while curr_start <= num_seconds - cfg.segment_len:
            curr_end = min(curr_start + 2 * cfg.segment_len, num_seconds)
            segment = signal[int(curr_start * cfg.sampling_rate):int(curr_end * cfg.sampling_rate)]
            spec = self._get_raw_spectrogram(segment)

            # look for a sound in top 3/4 of frequency range
            found, sound_start = self._find_sound(spec, sound_factor, freq_cutoff, cfg.spec_height)
            sound_start /= samples_per_sec # convert from samples to seconds

            if found:
                offsets.append(curr_start + sound_start)
                curr_start += sound_start + cfg.segment_len
            elif keep_empty:
                offsets.append(curr_start)
                curr_start += cfg.segment_len
            else:
                # look for a sound in bottom 1/4 of frequency range
                found, sound_start = self._find_sound(spec, sound_factor, 0, freq_cutoff)
                sound_start /= samples_per_sec # convert from samples to seconds
                if found:
                    offsets.append(curr_start + sound_start)
                    curr_start += sound_start + cfg.segment_len
                else:
                    curr_start += cfg.segment_len

        return offsets

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
    # you have to call load() before calling this
    def get_spectrograms(self, offsets, segment_len=cfg.segment_len, spec_exponent=cfg.spec_exponent, multi_spec=False):
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
                block = self._get_raw_spectrogram(self.signal[start:start+length])
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
            if info['streams'][0]['channels'] == 1:
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
                print(f'Caught exception in audio load: {tokens[-2]}')
            else:
                print(f'Caught exception in audio load')

        logging.debug('Done loading audio file')
        return self.signal, cfg.sampling_rate
