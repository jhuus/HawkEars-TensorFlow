# Select and shuffle a random subset of available data, and apply data augmentation techniques.

import random
import sys

import numpy as np
import skimage
from skimage import filters

from core import audio
from core import config as cfg
from core import util
from core import plot

CACHE_LEN = 1000   # cache this many noise specs for performance

# indexes of augmentation types
BLUR_INDEX = 0      # so self.probs[0] refers to blur
FADE_INDEX = 1
WHITE_NOISE_INDEX = 2
PINK_NOISE_INDEX = 3
REAL_NOISE_INDEX = 4
SHIFT_INDEX = 5
SPECKLE_INDEX = 6

class DataGenerator():
    def __init__(self, db, x_train, y_train, train_class):
        self.audio = audio.Audio()
        self.x_train = x_train
        self.y_train = y_train
        self.train_class = train_class

        if cfg.low_noise_detector:
            self.spec_height = cfg.lnd_spec_height
            # don't add noise when training low noise detector
            freqs = np.array([cfg.blur_freq, cfg.fade_freq, 0, 0, 0, cfg.shift_freq, cfg.speckle_freq])
        else:
            self.spec_height = cfg.spec_height
            freqs = np.array([cfg.blur_freq, cfg.fade_freq, cfg.white_noise_freq, cfg.pink_noise_freq,
                cfg.real_noise_freq, cfg.shift_freq, cfg.speckle_freq])

        self.indices = np.arange(y_train.shape[0])
        if cfg.augmentation:
            # convert relative frequencies to probability ranges in [0, 1]
            sum = np.sum(freqs)
            probs = freqs / sum
            self.probs = np.zeros(SPECKLE_INDEX + 1)
            self.probs[0] = probs[0]
            for i in range(1, SPECKLE_INDEX + 1):
                self.probs[i] = self.probs[i - 1] + probs[i]

            if not cfg.low_noise_detector:
                # create some white noise
                self.white_noise = np.zeros((CACHE_LEN, self.spec_height, cfg.spec_width, 1))
                for i in range(CACHE_LEN):
                    self.white_noise[i] = skimage.util.random_noise(self.white_noise[i], mode='gaussian', seed=cfg.seed, var=cfg.noise_variance, clip=True)
                    self.white_noise[i] /= np.max(self.white_noise[i]) # scale so max value is 1

                # create some pink noise
                self.pink_noise = np.zeros((CACHE_LEN, self.spec_height, cfg.spec_width, 1))
                for i in range(CACHE_LEN):
                    self.pink_noise[i] = self.audio.pink_noise()

                # get some noise spectrograms from the database
                results = db.get_spectrogram_by_subcat_name('Noise')
                self.real_noise = np.zeros((len(results), cfg.spec_height, cfg.spec_width, 1))
                for i, r in enumerate(results):
                    self.real_noise[i] = util.expand_spectrogram(r.value)

            self.speckle = np.zeros((CACHE_LEN, self.spec_height, cfg.spec_width, 1))
            for i in range(CACHE_LEN):
                self.speckle[i] = skimage.util.random_noise(self.speckle[i], mode='gaussian', seed=cfg.seed, var=cfg.speckle_variance, clip=True)

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = util.expand_spectrogram(self.x_train[id], low_noise_detector=cfg.low_noise_detector)
            label = self.y_train[id].astype(np.float32)

            if cfg.augmentation and cfg.multi_label and not cfg.low_noise_detector and self.train_class[id] != 'Noise':
                prob = random.uniform(0, 1)

                if prob < cfg.prob_merge:
                    spec, label = self.merge_specs(spec, label, self.train_class[id])

            if cfg.augmentation:
                prob = random.uniform(0, 1)
                if prob < cfg.prob_aug:
                    prob = random.uniform(0, 1)

                    # it's very important to check these in this order
                    if prob < self.probs[BLUR_INDEX]:
                        spec = self._blur(spec)
                    elif prob < self.probs[FADE_INDEX]:
                        spec = self._fade(spec)
                    elif prob < self.probs[WHITE_NOISE_INDEX]:
                        spec = self._add_noise(spec, self.white_noise)
                    elif prob < self.probs[PINK_NOISE_INDEX]:
                        spec = self._add_noise(spec, self.pink_noise)
                    elif prob < self.probs[REAL_NOISE_INDEX]:
                        spec = self._add_noise(spec, self.real_noise)
                    elif prob < self.probs[SHIFT_INDEX]:
                        spec = self._shift_horizontal(spec)
                    else:
                        spec = self._speckle(spec)

            # reduce values slightly
            spec *= random.uniform(cfg.min_mult, cfg.max_mult)

            yield (spec.astype(np.float32), label)

    # pick a random spectrogram and merge it with the given one
    def merge_specs(self, spec, label, class_name):
        index = random.randint(0, len(self.indices) - 1)
        other_id = self.indices[index]

        # loop until we get a different class that is not noise
        while self.train_class == 'Noise' or self.train_class == class_name:
            index = random.randint(0, len(self.indices) - 1)
            other_id = self.indices[index]

        other_spec = util.expand_spectrogram(self.x_train[other_id])
        spec += other_spec
        spec = spec.clip(0, 1)
        label += self.y_train[other_id].astype(np.float32)
        return spec, label

    # add noise to the spectrogram
    def _add_noise(self, spec, noise):
        index = random.randint(0, len(noise) - 1)
        spec = spec + noise[index] * random.uniform(cfg.noise_min, cfg.noise_max)
        spec /= np.max(spec)
        spec = spec.clip(0, 1)

        return spec

    # blur the spectrogram (larger values of sigma lead to more blurring)
    def _blur(self, spec, min_sigma=0.1, max_sigma=1.0):
        sigma = random.uniform(min_sigma, max_sigma)
        spec = skimage.filters.gaussian(spec, sigma=sigma)

        # renormalize to [0, 1]
        max = spec.max()
        if max > 0:
            spec = spec / max

        return spec

    # fade the spectrogram (smaller factors and larger min_vals lead to more fading);
    # defaults don't have a big visible effect but do fade values a little, and it's
    # important to preserve very faint spectrograms
    def _fade(self, spec):
        factor = random.uniform(cfg.min_fade_factor, cfg.max_fade_factor)
        spec *= factor
        spec[spec < cfg.min_fade_val] = 0 # clear values near zero
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
            max_shift_left = cfg.max_shift

        # detect right-shifted spectrograms, so we don't shift further right
        right_part = spec[:self.spec_height, cfg.spec_width - 10:]
        num_pixels = (right_part > 0.05).sum()
        if num_pixels > 300:
            max_shift_right = 0
        else:
            max_shift_right = cfg.max_shift

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
