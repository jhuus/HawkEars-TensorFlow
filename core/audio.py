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
adjust_mel_amplitude = [
    0.026919,0.027495,0.028521,0.034298,0.031367,0.031291,0.034798,0.034513,0.032859,0.038978,0.032325,0.042106,0.036139,0.041577,0.037585,0.037485,
    0.043633,0.042630,0.042206,0.043395,0.045717,0.046398,0.046903,0.047681,0.048802,0.050954,0.052144,0.053238,0.056013,0.057329,0.059150,0.063504,
    0.059934,0.065076,0.064353,0.063853,0.069758,0.073321,0.069831,0.069919,0.074158,0.079379,0.079901,0.082265,0.086648,0.087482,0.087969,0.087006,
    0.086812,0.091648,0.095205,0.097862,0.102273,0.102333,0.105423,0.108270,0.107726,0.112535,0.113461,0.116392,0.124740,0.124153,0.124960,0.131739,
    0.132585,0.138153,0.143185,0.146503,0.144147,0.153113,0.155650,0.162229,0.168459,0.170942,0.174179,0.174325,0.179911,0.184171,0.192862,0.195626,
    0.202218,0.202807,0.210233,0.214447,0.215467,0.222824,0.230259,0.234748,0.241555,0.250036,0.255858,0.259823,0.262786,0.268640,0.276871,0.287395,
    0.295998,0.307393,0.312396,0.317129,0.324253,0.333895,0.334569,0.347561,0.357874,0.375411,0.376225,0.389205,0.402477,0.401829,0.416931,0.427271,
    0.437032,0.446443,0.457271,0.472060,0.483338,0.495797,0.498201,0.511688,0.527619,0.544980,0.557429,0.575573,0.575069,0.604038,0.610146,0.630232,
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
            if cfg.mel_amplitude_adjustment and cfg.spec_height == len(adjust_mel_amplitude):
                spec = (spec.T / adjust_mel_amplitude).T # compensate for how mel scaling increases high frequency amplitudes
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
    def get_spectrograms(self, offsets, seconds=cfg.segment_len, spec_exponent=cfg.spec_exponent, multi_spec=False):
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
            start = 0
            block_length = cfg.spec_block_seconds * cfg.sampling_rate
            spec_width_per_sec
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
                    specs.append(spectrogram[:, int(offset * spec_width_per_sec) : int((offset + cfg.segment_len) * spec_width_per_sec)])
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
