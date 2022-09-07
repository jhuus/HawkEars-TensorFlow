# Given a list of spectrograms, select the ones that have "loud low frequencies".
# For those, call a neural network to decide if the lowest frequencies contain
# noise or other low frequency sounds that we don't want to treat as noise.
# If it's noise, the caller will reduce the low frequency amplitudes so they don't push other
# sounds too far into the background during normalization.

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = no info, 2 = no warnings, 3 = no errors
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from core import config as cfg

# using a different spectrogram array length every time triggers a Tensorflow warning about
# tf.function retracing, so use a fixed length and grow it when needed
LENGTH_INCR = 100

class LowNoiseDetector:
    def __init__(self, path_prefix):
        self.model = keras.models.load_model(f'{path_prefix}{cfg.low_noise_ckpt_path}', compile=False)
        self.specs = np.zeros((LENGTH_INCR, cfg.low_noise_spec_height, cfg.spec_width, 1))

    # return a list containing True if the corresponding spectrogram contains low frequency noise
    def check_for_noise(self, specs, low_idx=30, high_idx=40, low_mult=2.0, min_confidence=0.7):
        # initialize return values
        ret_vals = [False for i in range(len(specs))]
        high_maxes = [0 for i in range(len(specs))]

        # get the indexes of the ones with loud low frequencies
        indexes = []
        for i in range(len(specs)):
            spec = specs[i]
            if spec is None:
                continue

            low_max = np.max(spec[:low_idx,:])
            high_max = np.max(spec[high_idx:,:])
            high_maxes[i] = high_max
            if high_max > 0 and low_max > low_mult * high_max:
                indexes.append(i)

        if len(indexes) == 0:
            return ret_vals, high_maxes

        # copy relevant spectrograms into a numpy array and call the neural network;
        # first grow the array if needed
        while len(indexes) > self.specs.shape[0]:
            old_len = self.specs.shape[0]
            self.specs = np.zeros((old_len + LENGTH_INCR, cfg.low_noise_spec_height, cfg.spec_width, 1))

        for i in range(len(indexes)):
            self.specs[i, :, :, 0] = specs[indexes[i]][:cfg.low_noise_spec_height,:]

            # normalize values between 0 and 1
            max = self.specs[i].max()
            if max > 0:
                self.specs[i] /= max

        predictions = self.model.predict(self.specs)

        # update return values based on predictions
        for i in range(len(indexes)):
            # be sure before saying it's noise; if unsure, it's better to say no
            if predictions[i][0] > min_confidence:
                ret_vals[indexes[i]] = True

        return ret_vals, high_maxes
