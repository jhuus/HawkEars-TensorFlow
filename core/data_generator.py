# Select and shuffle a random subset of available data, and apply data augmentation techniques.

import logging
import random
import sys

import numpy as np
import skimage
from skimage import filters

from core import audio
from core import config as cfg
from core import util
from core import plot

CACHE_LEN = 1000   # cache this many white noise spectrograms for performance

# indexes of augmentation types
WHITE_NOISE_INDEX = 0
SHIFT_INDEX = 1
SPECKLE_INDEX = 2

class DataGenerator():
    def __init__(self, db, x_train, y_train, train_class):
        self.audio = audio.Audio()
        self.x_train = x_train
        self.y_train = y_train
        self.train_class = train_class
        self.set_probabilities()

        self.indices = np.arange(y_train.shape[0])
        if cfg.augmentation:
            # create some white noise
            self.white_noise = np.zeros((CACHE_LEN, cfg.spec_height, cfg.spec_width, 1))
            for i in range(CACHE_LEN):
                self.white_noise[i] = self._get_white_noise(cfg.white_noise_variance)

            self.speckle = np.zeros((CACHE_LEN, cfg.spec_height, cfg.spec_width, 1))
            for i in range(CACHE_LEN):
                self.speckle[i] = self._get_white_noise(cfg.speckle_variance)

    # get relative weights of augmentation types and convert to probability ranges in [0, 1]
    def set_probabilities(self):
        weights = np.array([cfg.white_noise_weight, cfg.shift_weight, cfg.speckle_weight])

        sum = np.sum(weights)
        probs = weights / sum
        self.probs = np.zeros(SPECKLE_INDEX + 1)
        self.probs[0] = probs[0]
        for i in range(1, SPECKLE_INDEX + 1):
            self.probs[i] = self.probs[i - 1] + probs[i]

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = util.expand_spectrogram(self.x_train[id])
            label = self.y_train[id].astype(np.float32)

            other_id = None
            if cfg.augmentation and cfg.multi_label and self.train_class[id] != 'Noise':
                prob = random.uniform(0, 1)

                if prob < cfg.prob_merge:
                    spec, label = self._merge_specs(spec, label, self.train_class[id])
                    spec = self._normalize_spec(spec)

            if cfg.augmentation:
                prob = random.uniform(0, 1)
                if prob < cfg.prob_aug:
                    prob = random.uniform(0, 1)

                    # it's very important to check these in this order
                    if prob < self.probs[WHITE_NOISE_INDEX]:
                        spec = self._add_white_noise(spec)
                    elif prob < self.probs[SHIFT_INDEX]:
                        spec = self._shift_horizontal(spec)
                    else:
                        spec = self._speckle(spec)

                    spec = self._normalize_spec(spec)

                # reduce the max value from 1
                spec *= random.uniform(cfg.min_fade, cfg.max_fade)

            yield (spec.astype(np.float32), label)

    # add white noise to the spectrogram
    def _add_white_noise(self, spec):
        index = random.randint(0, len(self.white_noise) - 1)
        spec += self.white_noise[index]
        return spec

    # return a white noise spectrogram with the given variance
    def _get_white_noise(self, variance):
        white_noise = np.zeros((1, cfg.spec_height, cfg.spec_width, 1))
        white_noise[0] = 1 + skimage.util.random_noise(white_noise[0], mode='gaussian', var=variance, clip=False)
        white_noise[0] -= np.min(white_noise) # set min = 0
        white_noise[0] /= np.max(white_noise) # set max = 1
        return white_noise[0]

    # pick a random spectrogram and merge it with the given one
    def _merge_specs(self, spec, label, class_name):
        index = random.randint(0, len(self.indices) - 1)
        other_id = self.indices[index]

        # loop until we get a different class
        while self.train_class == class_name:
            index = random.randint(0, len(self.indices) - 1)
            other_id = self.indices[index]

        other_spec = util.expand_spectrogram(self.x_train[other_id])
        spec += other_spec
        label += self.y_train[other_id].astype(np.float32)
        return spec, label

    # normalize so max value is 1
    def _normalize_spec(self, spec):
        max = spec.max()
        if max > 0:
            spec = spec / max

        return spec

    # perform a random horizontal shift of the spectrogram, using simple heuristics
    # to avoid shifting the important part of the signal past the edge
    def _shift_horizontal(self, spec):
        # detect left-shifted spectrograms, so we don't shift further left
        left_part = spec[:cfg.spec_height, :10]
        num_pixels = (left_part > 0.05).sum()
        if num_pixels > 300:
            max_shift_left = 0
        else:
            max_shift_left = cfg.max_shift

        # detect right-shifted spectrograms, so we don't shift further right
        right_part = spec[:cfg.spec_height, cfg.spec_width - 10:]
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

    # add a copy multiplied by random pixels (larger variances lead to more speckling)
    def _speckle(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec += spec * self.speckle[index]
        return spec
