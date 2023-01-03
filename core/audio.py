# Audio processing, especially extracting and returning spectrograms.

import logging
import random

import colorednoise as cn
import cv2
import ffmpeg
import numpy as np
import scipy
import tensorflow as tf

from . import low_noise_detector
from core import config as cfg
from core import plot

class Audio:
    def __init__(self, path_prefix=''):
        self.have_signal = False
        self.signal = None
        self.path_prefix = path_prefix
        self.low_noise_detector = None

    # width of spectrogram is determined by input signal length, and height = cfg.spec_height
    def _get_raw_spectrogram(self, signal):
        s = tf.signal.stft(signals=signal, frame_length=cfg.win_length, frame_step=cfg.hop_length, fft_length=cfg.win_length, pad_end=True)
        spec = tf.cast(tf.abs(s), tf.float32)

        # clip frequencies above max_freq
        num_freqs = spec.shape[1]
        clip_idx = int(2 * spec.shape[1] * cfg.max_freq / cfg.sampling_rate)
        spec = spec[:, :clip_idx]

        if cfg.mel_scale:
            # mel spectrogram
            num_spectrogram_bins = int(spec.shape[-1])
            linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                cfg.spec_height, num_spectrogram_bins, cfg.sampling_rate, cfg.min_freq, cfg.sampling_rate // 2)
            mel = tf.tensordot(spec, linear_to_mel_matrix, 1)
            mel.set_shape(spec.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
            spec = np.transpose(mel)
        else:
            # linear frequency scale (use only for plotting spectrograms)
            spec = cv2.resize(spec.numpy(), dsize=(cfg.spec_height, spec.shape[0]), interpolation=cv2.INTER_CUBIC)
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

    # use a neural net to identify low frequency noise;
    # if found, attenuate it to bring out other sounds
    def _dampen_low_noise(self, specs):
        if self.low_noise_detector is None:
            self.low_noise_detector = low_noise_detector.LowNoiseDetector(self.path_prefix)

        is_noise = self.low_noise_detector.check_for_noise(specs)

        for i in range(len(specs)):
            if is_noise[i]:
                # there are loud sounds in the low frequencies, and it's noise;
                # dampen it so quieter high frequency sounds don't disappear during normalization;
                # i.e. apply a very simple sort of high-pass filter heuristic
                for row in range(cfg.lnd_spec_height):
                    row_max = np.max(specs[i][row:row+1,:])
                    specs[i][row:row+1,:] *= row / cfg.lnd_spec_height

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

    # return a pink noise spectrogram with values in range [0, 1]
    def pink_noise(self):
        beta = random.uniform(1.2, 1.6)
        samples = cfg.segment_len * cfg.sampling_rate
        segment = cn.powerlaw_psd_gaussian(beta, samples)
        spec = self._get_raw_spectrogram(segment)
        spec = spec / spec.max() # normalize to [0, 1]
        return spec.reshape((cfg.spec_height, cfg.spec_width, 1))

    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this
    def get_spectrograms(self, offsets, seconds=cfg.segment_len, multi_spec=False, dampen_low_noise=False, low_noise_detector=False):
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
        if cfg.spec_exponent != 1:
            for i in range(len(specs)):
                specs[i] = specs[i] ** cfg.spec_exponent

        if low_noise_detector:
            for i in range(len(specs)):
                # return only the lowest frequencies for binary classifier spectrograms
                specs[i] = specs[i][:cfg.lnd_spec_height, :]
        else:
            if dampen_low_noise:
                self._dampen_low_noise(specs)
                self._normalize(specs)

        return specs

    def signal_len(self):
        return len(self.signal) if self.have_signal else 0

    def load(self, path, keep_bytes=False):
        self.have_signal = False
        self.signal = None
        spectrogram = None

        try:
            bytes, _ = (ffmpeg
                .input(path)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{cfg.sampling_rate}')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True))

            # convert byte array to float array, and then to a numpy array
            scale = 1.0 / float(1 << ((16) - 1))
            self.signal = scale * np.frombuffer(bytes, '<i2').astype(np.float32)
            self.have_signal = True

            if keep_bytes:
                self.bytes = bytes # when we want the raw audio, e.g. to write a segment to a wav file

        except ffmpeg.Error as e:
            tokens = e.stderr.decode().split('\n')
            if len(tokens) >= 2:
                print(f'Caught exception in audio load: {tokens[-2]}')
            else:
                print(f'Caught exception in audio load')

        logging.debug('Done loading audio file')
        return self.signal, cfg.sampling_rate
