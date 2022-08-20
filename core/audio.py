# Audio processing, especially extracting and returning 3-second spectrograms.
# This code was copied from https://github.com/kahst/BirdNET and then substantially modified.

import logging
import random

import colorednoise as cn
import ffmpeg
import numpy as np
import scipy
import scipy.signal

from . import binary_classifier
from core import constants

class Audio:
    def __init__(self, path_prefix=''):
        self.cache = {}
        self.have_signal = False
        self.binary_classifier = binary_classifier.BinaryClassifier(path_prefix)

    def _apply_bandpass_filter(self, sig):
        wn = np.array([constants.FMIN, constants.FMAX]) / (constants.SAMPLING_RATE / 2.0)
        order = 4
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')
        return scipy.signal.sosfiltfilt(filter_sos, sig)
        
    def _get_mel_filterbanks(self, num_banks, f_vec, dtype=np.float32):
        '''
        An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
        an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
        flooring down the bin indices.
        '''

        # filterbank already in cache?
        fname = 'mel_' + str(num_banks) + '_' + str(constants.FMIN) + '_' + str(constants.FMAX)
        if not fname in self.cache:
            
            # break frequency and scaling factor (smaller f_break values increase the scaling effect)
            A = 4581.0
            f_break = 1625.0

            # convert Hz to mel
            freq_extents_mel = A * np.log10(1 + np.asarray([constants.FMIN, constants.FMAX], dtype=dtype) / f_break)

            # compute points evenly spaced in mels
            melpoints = np.linspace(freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype)

            # convert mels to Hz
            banks_ends = (f_break * (10 ** (melpoints / A) - 1))

            filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)
            for bank_idx in range(1, num_banks+1):
                # points in the first half of the triangle
                mask = np.logical_and(f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx])
                filterbank[mask, bank_idx-1] = (f_vec[mask] - banks_ends[bank_idx - 1]) / \
                    (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

                # points in the second half of the triangle
                mask = np.logical_and(f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx+1])
                filterbank[mask, bank_idx-1] = (banks_ends[bank_idx + 1] - f_vec[mask]) / \
                    (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

            # scale and normalize, so that all the triangles do not have same height and the gain gets adjusted appropriately.
            temp = filterbank.sum(axis=0)
            non_zero_mask = temp > 0
            filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

            self.cache[fname] = (filterbank, banks_ends[1:-1])

        return self.cache[fname][0], self.cache[fname][1]

    def _get_raw_spectrogram(self, signal, shape):
        # compute overlap
        hop_len = int(len(signal) / (shape[1] - 1)) 
        n_fft = constants.WIN_LEN

        # ensure output width is 384 (or 385 if can't get 384)
        overlap_hash = {256: -87, 512: 168, 768: 426, 1024: 683, 1576: 1236, 2048: 1709, 4096: 3763}
        win_overlap = overlap_hash[constants.WIN_LEN]

        f, t, spec = scipy.signal.spectrogram(signal,
                                            fs=constants.SAMPLING_RATE,
                                            window=scipy.signal.windows.hann(constants.WIN_LEN),
                                            nperseg=constants.WIN_LEN,
                                            noverlap=win_overlap,
                                            nfft=n_fft,
                                            detrend=False,
                                            mode='magnitude')

        # determine the indices of where to clip the spec
        valid_f_idx_start = f.searchsorted(constants.FMIN, side='left')
        valid_f_idx_end = f.searchsorted(constants.FMAX, side='right') - 1

        spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
        
        # get mel filter banks
        mel_filterbank, mel_f = self._get_mel_filterbanks(shape[0], f, dtype=spec.dtype)

        # clip to non-zero range so that unnecessary multiplications can be avoided
        mel_filterbank = mel_filterbank[valid_f_idx_start:(valid_f_idx_end + 1), :]

        # clip the spec representation and apply the mel filterbank;
        # due to the nature of np.dot(), the spec needs to be transposed prior, and reverted after
        spec = np.dot(spec, mel_filterbank)
        spec = np.transpose(spec, [1, 0]) 
        spec = spec[:shape[0], :shape[1]]
        
        if spec.shape[1] == shape[1] - 1:
            # I don't know why this happens occasionally, but fix it here
            temp = spec
            spec = np.zeros((shape[0], shape[1]))
            spec[:,:-1] = temp
        
        return spec
        
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

        if found and first_sound_sample < constants.SPEC_WIDTH:
            # we found sound in the first segment, so try to center it;
            # scan backwards from end of potential segment until non-silence
            curr_offset = first_sound_sample
            end_offset = min(curr_offset + constants.SPEC_WIDTH - 1, spec.shape[0] - 1)
            
            while sound[end_offset] == 0:
                end_offset -= 1

            # determine padding and back up from curr_offset to center the sound
            sound_len = end_offset - curr_offset + 1
            total_pad_len = constants.SPEC_WIDTH - sound_len
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

        num_seconds = signal.shape[0] / constants.SAMPLING_RATE
        samples_per_sec = constants.SPEC_WIDTH / constants.SEGMENT_LEN
        freq_cutoff = int(constants.SPEC_HEIGHT / 4) # dividing line between "low" and "high" frequencies
        
        # process in chunks, where each is as wide as two spectrograms
        curr_start = 0
        offsets = []
        while curr_start <= num_seconds - constants.SEGMENT_LEN:
            curr_end = min(curr_start + 2 * constants.SEGMENT_LEN, num_seconds)
            segment = signal[int(curr_start * constants.SAMPLING_RATE):int(curr_end * constants.SAMPLING_RATE)]
            shape=(constants.SPEC_HEIGHT, int(constants.SPEC_WIDTH * ((curr_end - curr_start) / constants.SEGMENT_LEN)))
            spec = self._get_raw_spectrogram(segment, shape)

            # look for a sound in top 3/4 of frequency range
            found, sound_start = self._find_sound(spec, sound_factor, freq_cutoff, constants.SPEC_HEIGHT)
            sound_start /= samples_per_sec # convert from samples to seconds
            
            if found:
                offsets.append(curr_start + sound_start)
                curr_start += sound_start + constants.SEGMENT_LEN
            elif keep_empty:
                offsets.append(curr_start)
                curr_start += constants.SEGMENT_LEN
            else:
                # look for a sound in bottom 1/4 of frequency range
                found, sound_start = self._find_sound(spec, sound_factor, 0, freq_cutoff)
                sound_start /= samples_per_sec # convert from samples to seconds
                if found:
                    offsets.append(curr_start + sound_start)
                    curr_start += sound_start + constants.SEGMENT_LEN
                else:
                    curr_start += constants.SEGMENT_LEN
                
        return offsets

    # return a pink noise spectrogram with values in range [0, 1]
    def pink_noise(self):
        beta = random.uniform(1.2, 1.6)
        samples = constants.SEGMENT_LEN * constants.SAMPLING_RATE
        segment = cn.powerlaw_psd_gaussian(beta, samples)
        shape = (constants.SPEC_HEIGHT, constants.SPEC_WIDTH)
        spec = self._get_raw_spectrogram(segment, shape)
        spec = spec / spec.max() # normalize to [0, 1]
        return spec.reshape((constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1))

    def dampen_low_noise(self, specs, low_idx=5, high_idx=15, low_mult=2.0):
        is_noise, high_max = self.binary_classifier.check_for_noise(specs, low_idx, high_idx, low_mult)

        for i in range(len(specs)):
            if is_noise[i]:
                # there are loud sounds in the low frequencies, and it's noise;
                # dampen it so quieter high frequency sounds don't disappear during normalization
                for row in range(high_idx):
                    row_max = np.max(specs[i][row:row+1,:])
                    if row_max > high_max[i]:
                        # after this, the loudest low frequency sound will be as loud as the loudest high frequency sound,
                        # and quieter low frequency sounds are affected less
                        specs[i][row:row+1,:] = (specs[i][row:row+1,:] ** 0.5) * (high_max[i] / (row_max ** 0.5))

    def update_levels(self, specs, exponent=0.8, row_factor=0.8):
        for i in range(len(specs)):
            # this brings out faint sounds, but also increases noise;
            # increasing exponent gives less noise but loses some faint sounds
            specs[i] = specs[i] ** exponent

            # for each frequency, subtract a multiple of the average amplitude
            if row_factor > 0:
                num_freqs = specs[i].shape[0]
                for j in range(num_freqs):
                    specs[i][j] -= row_factor * np.average(specs[i][j])

    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this
    def get_spectrograms(self, offsets, shape=(constants.SPEC_HEIGHT, constants.SPEC_WIDTH), seconds=constants.SEGMENT_LEN, binary_classifier=False,
                         check_noise=True, update_levels=True, low_idx=5, high_idx=15, low_mult=2.0, exponent=0.8, min_val=0, row_factor=0.8):
        if not self.have_signal:
            return None

        last_offset = (len(self.signal) / constants.SAMPLING_RATE) - constants.SEGMENT_LEN
        specs = []
        for offset in offsets:
            if offset > last_offset:
                # not enough time for this offset, so pad it
                self.signal = np.pad(self.signal, (0, constants.SEGMENT_LEN * constants.SAMPLING_RATE), 'constant', constant_values=(0, 0))
                
            segment = self.signal[int(offset*constants.SAMPLING_RATE):int((offset+seconds)*constants.SAMPLING_RATE)]
            specs.append(self._get_raw_spectrogram(segment, shape))

        if binary_classifier:
            for i in range(len(specs)):
                # return only the lowest frequencies for binary classifier spectrograms
                specs[i] = specs[i][:constants.BINARY_SPEC_HEIGHT, :]
        else:
            if check_noise:
                self.dampen_low_noise(specs, low_idx, high_idx, low_mult)
            
            if update_levels:
                self.update_levels(specs, exponent, row_factor)

        if update_levels:
            for i in range(len(specs)):
                # normalize values between 0 and 1
                max = specs[i].max()
                if max > 0:
                    specs[i] /= max
                    
                specs[i] = specs[i].clip(min_val, 1)

        logging.info('Done creating spectrograms')
        return specs
        
    def load(self, path, filter=True):
        try:
            # on some systems librosa calls audioread which calls gstreamer, which
            # does a poor job decoding mp3's, so use ffmpeg instead;
            # also ffmpeg issues fewer warnings and loads significantly faster
            bytes, _ = (ffmpeg
                .input(path)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=f'{constants.SAMPLING_RATE}')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True))

            # convert byte array to float array, and then to a numpy array
            scale = 1.0 / float(1 << ((16) - 1))
            fmt = "<i{:d}".format(2)
            floats = scale * np.frombuffer(bytes, fmt).astype(np.float32)
            signal = np.asarray(floats)
            self.have_signal = True
            
            if filter:
                self.signal = self._apply_bandpass_filter(signal)
            else:
                self.signal = signal
        except ffmpeg.Error as e:
            tokens = e.stderr.decode().split('\n')
            if len(tokens) >= 2:
                print(f'Caught exception in audio load: {tokens[-2]}')
            else:
                print(f'Caught exception in audio load')

            self.signal = None
            self.have_signal = False


        logging.info('Done loading audio file')
        return self.signal, constants.SAMPLING_RATE
