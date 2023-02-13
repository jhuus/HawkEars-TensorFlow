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

CACHE_LEN = 1000   # cache this many white noise spectrograms

class DataGenerator():
    def __init__(self, db, x_train, y_train, train_class):
        self.audio = audio.Audio()
        self.x_train = x_train
        self.y_train = y_train
        self.train_class = train_class

        self.indices = np.arange(y_train.shape[0])
        if cfg.augmentation:
            # create some white noise
            self.white_noise = np.zeros((CACHE_LEN, cfg.spec_height, cfg.spec_width, 1))
            for i in range(CACHE_LEN):
                variance = random.uniform(cfg.min_white_noise_variance, cfg.max_white_noise_variance)
                self.white_noise[i] = self._get_white_noise(variance)

            self.speckle = np.zeros((CACHE_LEN, cfg.spec_height, cfg.spec_width, 1))
            for i in range(CACHE_LEN):
                self.speckle[i] = self._get_white_noise(cfg.speckle_variance)

            # get some noise spectrograms from the database
            results = db.get_spectrogram_by_subcat_name('Noise')
            self.real_noise = np.zeros((len(results), cfg.spec_height, cfg.spec_width, 1))
            for i, r in enumerate(results):
                self.real_noise[i] = util.expand_spectrogram(r.value) * cfg.real_noise_factor

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
                    if prob < cfg.prob_speckle:
                        spec = self._speckle(spec)
                    elif prob < cfg.prob_real_noise:
                        spec = self._add_real_noise(spec)
                    else:
                        spec = self._add_white_noise(spec)

                    spec = self._normalize_spec(spec)

                # reduce the max value from 1
                spec *= random.uniform(cfg.min_fade, cfg.max_fade)

            yield (spec.astype(np.float32), label)

    # add white noise to the spectrogram
    def _add_white_noise(self, spec):
        index = random.randint(0, len(self.white_noise) - 1)
        spec += self.white_noise[index]
        return spec

    # add real noise to the spectrogram
    def _add_real_noise(self, spec):
        index = random.randint(0, len(self.real_noise) - 1)
        spec += self.real_noise[index]
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

    # add a copy multiplied by random pixels (larger variances lead to more speckling)
    def _speckle(self, spec):
        index = random.randint(0, CACHE_LEN - 1)
        spec += spec * self.speckle[index]
        return spec
