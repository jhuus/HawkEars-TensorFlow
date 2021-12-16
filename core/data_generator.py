# Select and shuffle a random subset of available data, and apply data augmentation techniques.
# I tried adding a separate thread to fill a queue of unzipped spectrograms,
# but it didn't actually improve performance.

import random
import sys

import numpy as np
import skimage
from skimage import filters

from core import audio
from core import constants
from core import util

MAX_CHANGES = 2            # max changes per spectrogram
MAX_SHIFT = 5              # max pixels for horizontal shift
PROB_CHANGES = [0.33, 0.66] # p < P[0] = 0 changes, else p < P[1] = 1 change, else 2 changes

# these probabilities affect the relative frequency of the augmentation types
PROB_BLUR = 0.25
PROB_FADE = 0.5
PROB_LOW = 0.25
PROB_NOISE = 1.0
PROB_SHIFT = 0.5
PROB_SPECKLE = 1.0

class DataGenerator():
    def __init__(self, x_train, y_train, seed=None, augmentation=True, binary_classifier=False):
        self.x_train = x_train
        self.y_train = y_train
        self.seed = seed
        self.augmentation = augmentation
        self.binary_classifier = binary_classifier
        
        if binary_classifier:
            self.spec_height = constants.BINARY_SPEC_HEIGHT
        else:
            self.spec_height = constants.SPEC_HEIGHT
        
        self.indices = np.arange(y_train.shape[0])
        if self.augmentation:
            # self.local_vars is used in _add_low_noise;
            # use exponentially more noise in the low frequencies
            self.local_vars = np.zeros((self.spec_height, constants.SPEC_WIDTH, 1))
            for row in range(0, self.spec_height):
                max_val = ((self.spec_height - row) ** 4 / self.spec_height ** 4) / 70
                for col in range(0, constants.SPEC_WIDTH):
                    self.local_vars[row, col, 0] = np.random.uniform(0.001, max_val)

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = util.expand_spectrogram(self.x_train[id], binary_classifier=self.binary_classifier)
            
            if self.augmentation:
                # detect left-shifted spectrograms, so we don't shift further left
                left_part = spec[:self.spec_height, :10]
                num_pixels = (left_part > 0.05).sum()
                if num_pixels > 300:
                    max_shift_left = 0
                else:
                    max_shift_left = MAX_SHIFT
                
                # detect right-shifted spectrograms, so we don't shift further right
                right_part = spec[:self.spec_height, constants.SPEC_WIDTH - 10:]
                num_pixels = (right_part > 0.05).sum()
                if num_pixels > 300:
                    max_shift_right = 0
                else:
                    max_shift_right = MAX_SHIFT

                # decide the number of changes to apply to this spectrogram
                changes_todo = MAX_CHANGES
                prob = np.random.uniform(0, 1)
                for j in range(MAX_CHANGES):
                    if prob < PROB_CHANGES[j]:
                        changes_todo = j
                        break

                # ensure we don't apply the same change twice
                done_blur = False
                done_fade = False
                done_noise = False # includes regular and low-frequency noise, so we don't apply both to the same spectrogram
                done_shift = False
                done_speckle = False
                
                # loop, applying changes (data augmentations)
                changes = ['blur', 'fade', 'low', 'noise', 'shift', 'speckle']
                changes_done = 0
                while changes_done < changes_todo:
                    change_index = random.randint(0, len(changes) - 1) # pick one at random
                    change = changes[change_index]
                    prob = np.random.uniform(0, 1)
                    
                    if change == 'none':
                        changes_done += 1
                    elif change == 'blur' and not done_blur and prob < PROB_BLUR:
                        spec = self._blur(spec)
                        done_blur = True
                        changes_done += 1
                    elif change == 'fade' and not done_fade and prob < PROB_FADE:
                        spec = self._fade(spec)
                        done_fade = True
                        changes_done += 1
                    elif change == 'low' and not done_noise and prob < PROB_LOW:
                        spec = self._add_low_noise(spec)
                        done_noise = True
                        changes_done += 1
                    elif change == 'noise' and not done_noise and prob < PROB_NOISE:
                        spec = self._add_noise(spec)
                        done_noise = True
                        changes_done += 1
                    elif change == 'shift' and not done_shift and prob < PROB_SHIFT:
                        spec = self._shift_horizontal(spec, max_shift_left, max_shift_right)
                        done_shift = True
                        changes_done += 1
                    elif change == 'speckle' and not done_speckle and prob < PROB_SPECKLE:
                        spec = self._speckle(spec)
                        done_speckle = True
                        changes_done += 1
            
            yield (spec.astype(np.float32), self.y_train[id].astype(np.float32))
    
    # add low frequency noise to the spectrogram
    def _add_low_noise(self, spec):
        spec = skimage.util.random_noise(spec, mode='localvar', seed=self.seed, local_vars=self.local_vars, clip=True)
        return spec

    # add random noise to the spectrogram (larger variances lead to more noise)
    def _add_noise(self, spec, variance=0.001):
        spec = skimage.util.random_noise(spec, mode='gaussian', seed=self.seed, var=variance, clip=True)
        return spec

    # blur the spectrogram (larger values of sigma lead to more blurring)
    def _blur(self, spec, min_sigma=0.1, max_sigma=1):
        sigma = np.random.uniform(min_sigma, max_sigma)
        spec = skimage.filters.gaussian(spec, sigma=sigma, multichannel=False)
        return spec
        
    # fade the spectrogram (smaller factors and larger min_vals lead to more fading);
    # defaults don't have a big visible effect but do fade values a little, and it's
    # important to preserve very faint spectrograms
    def _fade(self, spec, min_factor=0.8, max_factor=0.99, min_val=0.06):
        factor = np.random.uniform(min_factor, max_factor)
        spec *= factor
        spec[spec < min_val] = 0 # clear values near zero
        spec *= 1/factor # rescale so max = 1
        spec = np.clip(spec, 0, 1) # just to be safe
        return spec
                
    # perform a random horizontal shift of the spectrogram
    def _shift_horizontal(self, spec, max_shift_left, max_shift_right):
        if max_shift_left == 0 and max_shift_right == 0:
            return spec
        
        pixels = random.randint(-max_shift_left, max_shift_right)
        spec = np.roll(spec, shift=pixels, axis=1)
        return spec

    # multiply by random pixels (larger variances lead to more speckling)
    def _speckle(self, spec, variance=0.01):
        spec = skimage.util.random_noise(spec, mode='speckle', seed=self.seed, var=variance, clip=True)
        return spec
