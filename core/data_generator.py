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

PROB_AUG = 0.55    # probability of augmentation
CACHE_LEN = 1000   # cache this many noise specs for performance

MAX_SHIFT = 5      # max pixels for horizontal shift
NOISE_VARIANCE = 0.0015 # larger variances lead to more noise
SPECKLE_VARIANCE = .009

# relative frequencies of the augmentation types
BLUR_INDEX = 0      # so FREQS[0] is relative frequency of blur
FADE_INDEX = 1
LOW_INDEX = 2
NOISE_INDEX = 3
SHIFT_INDEX = 4
SPECKLE_INDEX = 5

FREQS = [0.25, 0.5, 0.2, 1.0, 0.5, 0.9]

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
            # convert relative frequencies to probability ranges in [0, 1]
            freqs = np.array(FREQS)
            sum = np.sum(freqs)
            probs = freqs / sum
            self.probs = np.zeros(SPECKLE_INDEX + 1)
            self.probs[0] = probs[0]
            for i in range(1, SPECKLE_INDEX + 1):
                self.probs[i] = self.probs[i - 1] + probs[i]

            # self.local_vars is used in _add_low_noise;
            # use exponentially more noise in the low frequencies
            self.local_vars = np.zeros((self.spec_height, constants.SPEC_WIDTH, 1))
            for row in range(0, self.spec_height):
                max_val = ((self.spec_height - row) ** 4 / self.spec_height ** 4) / 70
                for col in range(0, constants.SPEC_WIDTH):
                    self.local_vars[row, col, 0] = np.random.uniform(0.001, max_val)

            # creating these here instead of during augmentation saves a lot of time
            self.noise = np.zeros((CACHE_LEN, self.spec_height, constants.SPEC_WIDTH, 1))
            for i in range(CACHE_LEN):
                self.noise[i] = skimage.util.random_noise(self.noise[i], mode='gaussian', seed=self.seed, var=NOISE_VARIANCE, clip=True)

            self.low_noise = np.zeros((CACHE_LEN, self.spec_height, constants.SPEC_WIDTH, 1))
            for i in range(CACHE_LEN):
                self.low_noise[i] = skimage.util.random_noise(self.low_noise[i], mode='localvar', seed=self.seed, local_vars=self.local_vars, clip=True)

            self.speckle = np.zeros((CACHE_LEN, self.spec_height, constants.SPEC_WIDTH, 1))
            for i in range(CACHE_LEN):
                self.speckle[i] = skimage.util.random_noise(self.speckle[i], mode='gaussian', seed=self.seed, var=SPECKLE_VARIANCE, clip=True)

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = util.expand_spectrogram(self.x_train[id], binary_classifier=self.binary_classifier)
            
            if self.augmentation:
                prob = np.random.uniform(0, 1)
                if prob < PROB_AUG:
                    prob = np.random.uniform(0, 1)

                    # it's very important to check these in this order
                    if prob < self.probs[BLUR_INDEX]:
                        spec = self._blur(spec)
                    elif prob < self.probs[FADE_INDEX]:
                        spec = self._fade(spec)
                    elif prob < self.probs[LOW_INDEX]:
                        spec = self._add_low_noise(spec)
                    elif prob < self.probs[NOISE_INDEX]:
                        spec = self._add_noise(spec)
                    elif prob < self.probs[SHIFT_INDEX]:
                        spec = self._shift_horizontal(spec)
                    else:
                        spec = self._speckle(spec)
            
            yield (spec.astype(np.float32), self.y_train[id].astype(np.float32))
    
    # add low frequency noise to the spectrogram
    def _add_low_noise(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec += self.low_noise[index]
        spec = spec.clip(0, 1)
        return spec

    # add random noise to the spectrogram
    def _add_noise(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec += self.noise[index]
        spec = spec.clip(0, 1)
        return spec

    # blur the spectrogram (larger values of sigma lead to more blurring)
    def _blur(self, spec, min_sigma=0.1, max_sigma=1.0):
        sigma = np.random.uniform(min_sigma, max_sigma)
        spec = skimage.filters.gaussian(spec, sigma=sigma, multichannel=False)
        return spec
        
    # fade the spectrogram (smaller factors and larger min_vals lead to more fading);
    # defaults don't have a big visible effect but do fade values a little, and it's
    # important to preserve very faint spectrograms
    def _fade(self, spec, min_factor=0.6, max_factor=0.99, min_val=0.06):
        factor = np.random.uniform(min_factor, max_factor)
        spec *= factor
        spec[spec < min_val] = 0 # clear values near zero
        spec *= 1/factor # rescale so max = 1
        spec = np.clip(spec, 0, 1) # just to be safe
        return spec
                
    # perform a random horizontal shift of the spectrogram
    def _shift_horizontal(self, spec):
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

        if max_shift_left == 0 and max_shift_right == 0:
            return spec
        
        pixels = random.randint(-max_shift_left, max_shift_right)
        spec = np.roll(spec, shift=pixels, axis=1)
        return spec

    # multiply by random pixels (larger variances lead to more speckling)
    def _speckle(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec = spec + spec * self.speckle[index]
        spec = spec.clip(0, 1)
        return spec
