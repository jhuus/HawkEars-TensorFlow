# Audio processing, especially extracting and returning 3-second spectrograms.
# This code was copied from https://github.com/kahst/BirdNET and then substantially modified.

import numpy as np
import librosa
import scipy
import math
import sys

from . import binary_classifier
from core import constants
from core import util

class Audio:
    def __init__(self, path_prefix=''):
        self.cache = {}
        self.have_signal = False
        self.binary_classifier = binary_classifier.BinaryClassifier(path_prefix)

    def _apply_bandpass_filter(self, sig, rate, fmin, fmax):
        wn = np.array([fmin, fmax]) / (rate / 2.0)
        order = 4
        filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')

        fname = 'bandpass_' + str(rate) + '_' + str(fmin) + '_' + str(fmax)
        self.cache[fname] = filter_sos

        return scipy.signal.sosfiltfilt(self.cache[fname], sig)
        
    def _get_mel_filterbanks(self, num_banks, fmin, fmax, f_vec, dtype=np.float32):
        '''
        An arguably better version of librosa's melfilterbanks wherein issues with "hard snapping" are avoided. Works with
        an existing vector of frequency bins, as returned from signal.spectrogram(), instead of recalculating them and
        flooring down the bin indices.
        '''

        # filterbank already in cache?
        fname = 'mel_' + str(num_banks) + '_' + str(fmin) + '_' + str(fmax)
        if not fname in self.cache:
            
            # break frequency and scaling factor (smaller f_break values increase the scaling effect)
            A = 4581.0
            f_break = 1625.0

            # convert Hz to mel
            freq_extents_mel = A * np.log10(1 + np.asarray([fmin, fmax], dtype=dtype) / f_break)

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

    def _get_raw_spectrogram(self, signal, rate, shape, win_len=768):
        # compute overlap
        hop_len = int(len(signal) / (shape[1] - 1)) 
        win_overlap = win_len - hop_len + 2
        n_fft = win_len

        # compute spectrogram
        f, t, spec = scipy.signal.spectrogram(signal,
                                              fs=rate,
                                              window=scipy.signal.windows.hann(win_len),
                                              nperseg=win_len,
                                              noverlap=win_overlap,
                                              nfft=n_fft,
                                              detrend=False,
                                              mode='magnitude')
                                              
        # determine the indices of where to clip the spec
        valid_f_idx_start = f.searchsorted(self.fmin, side='left')
        valid_f_idx_end = f.searchsorted(self.fmax, side='right') - 1

        spec = np.transpose(spec[valid_f_idx_start:(valid_f_idx_end + 1), :], [1, 0])
        
        # get mel filter banks
        mel_filterbank, mel_f = self._get_mel_filterbanks(shape[0], self.fmin, self.fmax, f, dtype=spec.dtype)

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
    def find_sounds(self, signal=None, rate=None, sound_factor=1.15, keep_empty=True):
        if signal is None:
            if self.have_signal:
                signal = self.signal
            else:
                return []
            
        if rate is None:
            rate = self.rate

        num_seconds = signal.shape[0] / self.rate
        samples_per_sec = constants.SPEC_WIDTH / constants.SEGMENT_LEN
        freq_cutoff = int(constants.SPEC_HEIGHT / 4) # dividing line between "low" and "high" frequencies
        
        # process in chunks, where each is as wide as two spectrograms
        curr_start = 0
        offsets = []
        while curr_start <= num_seconds - constants.SEGMENT_LEN:
            curr_end = min(curr_start + 2 * constants.SEGMENT_LEN, num_seconds)
            segment = signal[int(curr_start * rate):int(curr_end * rate)]
            shape=(constants.SPEC_HEIGHT, int(constants.SPEC_WIDTH * ((curr_end - curr_start) / constants.SEGMENT_LEN)))
            spec = self._get_raw_spectrogram(segment, self.rate, shape)

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

    # return a Gaussian noise spectrogram with values in [center, 1 - clip];
    # clip is selected randomly in [min_clip, max_clip] and larger clip values make it sparser 
    def noise(self, shape=(constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1), center=0, stdev=1, min_clip=0.05, max_clip=0.25):
        spec = np.random.normal(center, stdev, shape)
        spec = spec / np.max(spec)
        clip = np.random.uniform(min_clip, max_clip)
        spec = np.clip(spec, clip, 1)
        spec = spec - clip
        return spec
        
    # return list of spectrograms for the given offsets (i.e. starting points in seconds);
    # you have to call load() before calling this
    def get_spectrograms(self, offsets, shape=(constants.SPEC_HEIGHT, constants.SPEC_WIDTH), seconds=constants.SEGMENT_LEN, win_len=768, binary_classifier=False,
                         check_noise=True, low_idx=5, high_idx=15, low_mult=2.0, exponent=0.8, min_val=0, row_factor=0.8):
        if not self.have_signal:
            return None

        last_offset = (len(self.signal) / self.rate) - constants.SEGMENT_LEN
        specs = []
        for offset in offsets:
            if offset > last_offset:
                # not enough time for this offset, so pad it
                self.signal = np.pad(self.signal, (0, constants.SEGMENT_LEN), 'constant', constant_values=(0, 0))
                
            segment = self.signal[int(offset*self.rate):int((offset+seconds)*self.rate)]
            specs.append(self._get_raw_spectrogram(segment, self.rate, shape, win_len))

        if binary_classifier:
            for i in range(len(specs)):
                # return only the lowest frequencies for binary classifier spectrograms
                specs[i] = specs[i][:constants.BINARY_SPEC_HEIGHT, :]
        else:
            if check_noise:
                # calling this once for the list is much faster than calling it for each one separately
                is_noise, high_max = self.binary_classifier.check_for_noise(specs, low_idx, high_idx, low_mult)
            
            for i in range(len(specs)):
                if check_noise and is_noise[i]:
                    # there are loud sounds in the low frequencies, and it's noise;
                    # dampen it so quieter high frequency sounds don't disappear during normalization
                    for row in range(high_idx):
                        row_max = np.max(specs[i][row:row+1,:])
                        if row_max > high_max[i]:
                            specs[i][row:row+1,:] = (specs[i][row:row+1,:] ** 0.5) * (high_max[i] / (row_max ** 0.5))
                        else:
                            # this is for low frequency pops and crackles, so stop when it fades out
                            break

                # this brings out faint sounds, but also increases noise;
                # increasing exponent gives less noise but loses some faint sounds
                specs[i] = specs[i] ** exponent

                # for each frequency, subtract a multiple of the average amplitude
                if row_factor > 0:
                    num_freqs = specs[i].shape[0]
                    for j in range(num_freqs):
                        specs[i][j] -= row_factor * np.average(specs[i][j])

        for i in range(len(specs)):
            # normalize values between 0 and 1
            max = specs[i].max()
            if max > 0:
                specs[i] /= max
                
            specs[i] = specs[i].clip(min_val, 1)

        return specs
        
    def load(self, path, fmin=50, fmax=12000, filter=True):
        try:
            signal, self.rate = librosa.load(path, sr=44100, mono=True, res_type='kaiser_fast')
            self.fmin, self.fmax = (fmin, fmax)
            self.have_signal = True
            
            if filter:
                self.signal = self._apply_bandpass_filter(signal, self.rate, fmin, fmax)
            else:
                self.signal = signal
        except:
            self.signal = None
            self.rate, self.fmin, self.fmax = (1, 1, 1)
            self.have_signal = False
            
        return self.signal, self.rate
