# Select and shuffle a random subset of available data, and apply data augmentation techniques.

import random
import sys

import numpy as np
import skimage
from skimage import filters

from core import audio
from core import constants
from core import util
from core import plot

PROB_MERGE = 0.20  # probability of merging to train multi-label support
PROB_AUG = 0.55    # probability of augmentation
CACHE_LEN = 1000   # cache this many noise specs for performance

MAX_SHIFT = 5      # max pixels for horizontal shift
NOISE_VARIANCE = 0.0015 # larger variances lead to more noise
SPECKLE_VARIANCE = .009

# relative frequencies of the augmentation types
BLUR_INDEX = 0      # so FREQS[0] is relative frequency of blur
FADE_INDEX = 1
WHITE_NOISE_INDEX = 2
PINK_NOISE_INDEX = 3
REAL_NOISE_INDEX = 4
SHIFT_INDEX = 5
SPECKLE_INDEX = 6

FREQS = [0.25, 0.5, 1.0, 1.0, 0.5, 0.5, 0.9]

class DataGenerator():
    def __init__(self, db, x_train, y_train, seed=None, augmentation=True, binary_classifier=False, multilabel=False):
        self.audio = audio.Audio()
        self.x_train = x_train
        self.y_train = y_train
        self.seed = seed
        self.augmentation = augmentation
        self.binary_classifier = binary_classifier
        self.multilabel = multilabel
        
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

            # create some white noise
            self.white_noise = np.zeros((CACHE_LEN, self.spec_height, constants.SPEC_WIDTH, 1))
            for i in range(CACHE_LEN):
                self.white_noise[i] = skimage.util.random_noise(self.white_noise[i], mode='gaussian', seed=self.seed, var=NOISE_VARIANCE, clip=True)

            # create some pink noise
            self.pink_noise = np.zeros((CACHE_LEN, self.spec_height, constants.SPEC_WIDTH, 1))
            for i in range(CACHE_LEN):
                self.pink_noise[i] = self.audio.pink_noise() + self.white_noise[i]

            # get some noise spectrograms from the database
            results = db.get_spectrograms_by_name('Denoiser')
            self.real_noise = np.zeros((len(results), constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1))
            for i in range(len(self.real_noise)):
                self.real_noise[i] = util.expand_spectrogram(results[i][0])

            self.speckle = np.zeros((CACHE_LEN, self.spec_height, constants.SPEC_WIDTH, 1))
            for i in range(CACHE_LEN):
                self.speckle[i] = skimage.util.random_noise(self.speckle[i], mode='gaussian', seed=self.seed, var=SPECKLE_VARIANCE, clip=True)

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = util.expand_spectrogram(self.x_train[id], binary_classifier=self.binary_classifier)
            label = self.y_train[id].astype(np.float32)
            
            if self.multilabel and not self.binary_classifier:
                prob = random.uniform(0, 1)

                if prob < PROB_MERGE:
                    spec, label = self.merge_specs(spec, label)

            if self.augmentation:
                prob = random.uniform(0, 1)
                if prob < PROB_AUG:
                    prob = random.uniform(0, 1)

                    # it's very important to check these in this order
                    if prob < self.probs[BLUR_INDEX]:
                        spec = self._blur(spec)
                    elif prob < self.probs[FADE_INDEX]:
                        spec = self._fade(spec)
                    elif prob < self.probs[WHITE_NOISE_INDEX]:
                        spec = self._add_white_noise(spec)
                    elif prob < self.probs[PINK_NOISE_INDEX]:
                        spec = self._add_pink_noise(spec)
                    elif prob < self.probs[REAL_NOISE_INDEX]:
                        spec = self._add_real_noise(spec)
                    elif prob < self.probs[SHIFT_INDEX]:
                        spec = self._shift_horizontal(spec)
                    else:
                        spec = self._speckle(spec)

            yield (spec.astype(np.float32), label)
    
    # pick a random spectrogram and merge it with the given one
    def merge_specs(self, spec, label):
        index = random.randint(0, len(self.indices) - 1)
        other_id = self.indices[index]
        other_spec = util.expand_spectrogram(self.x_train[other_id])
        spec += other_spec
        spec = spec.clip(0, 1)
        label += self.y_train[other_id].astype(np.float32)
        return spec, label

    # add white noise to the spectrogram
    def _add_white_noise(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec += self.white_noise[index]
        spec = spec.clip(0, 1)
        return spec

    # add low frequency noise to the spectrogram
    def _add_pink_noise(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        noise_mult = random.uniform(1.0, 3.0) # > 0 causes original spectrogram to fade accordingly
        spec += self.pink_noise[index] * noise_mult
        spec /= np.max(spec)
        spec = spec.clip(0, 1)
        return spec

    # add real noise to the spectrogram
    def _add_real_noise(self, spec):
        index = random.randint(0, len(self.real_noise) - 1)
        noise_mult = random.uniform(0.3, 0.85)
        spec += self.real_noise[index] * noise_mult
        spec /= np.max(spec)
        spec = spec.clip(0, 1)
        return spec

    # blur the spectrogram (larger values of sigma lead to more blurring)
    def _blur(self, spec, min_sigma=0.1, max_sigma=1.0):
        sigma = random.uniform(min_sigma, max_sigma)
        spec = skimage.filters.gaussian(spec, sigma=sigma, multichannel=False)

        # renormalize to [0, 1]
        max = spec.max()
        if max > 0:
            spec = spec / max

        return spec
        
    # fade the spectrogram (smaller factors and larger min_vals lead to more fading);
    # defaults don't have a big visible effect but do fade values a little, and it's
    # important to preserve very faint spectrograms
    def _fade(self, spec, min_factor=0.6, max_factor=0.99, min_val=0.06):
        factor = random.uniform(min_factor, max_factor)
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
