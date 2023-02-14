# Species-specific handling during analysis / inference.
# To disable or add a handler, update self.handlers in the constructor below.

from types import SimpleNamespace
import numpy as np
from core import config as cfg

class Species_Handlers:
    def __init__(self, class_infos, offsets, raw_spectrograms):
        # update this dictionary to enable/disable handlers
        self.handlers = {
            'BWHA': self.check_soundalike,
            'PIGR': self.check_amplitude,
        }

        # handler parameters, so it's easy to use the same logic for multiple species
        self.check_amplitude_config = {
            'PIGR': SimpleNamespace(low_freq=.58, high_freq=.67, min_ratio=.15)
        }

        self.check_soundalike_config = {
            'BWHA': SimpleNamespace(soundalike_code='WTSP', min_prob=.25)
        }

        self.class_infos = {}
        for class_info in class_infos:
            self.class_infos[class_info.code] = class_info

        self.offsets = offsets
        self.raw_spectrograms = raw_spectrograms
        self.highest_amplitude = None

    # Handle cases where a faint vocalization is mistaken for another species.
    # For example, distant songs of American Robin and similar-sounding species are sometimes mistaken for Pine Grosbeak,
    # so we ignore Pine Grosbeak sounds that are too quiet.
    def check_amplitude(self, class_info):
        config = self.check_amplitude_config[class_info.code]
        low_index = int(config.low_freq * cfg.spec_height)   # bottom of frequency range
        high_index = int(config.high_freq * cfg.spec_height) # top of frequency range

        for i in range(len(class_info.probs)):
            # ignore if probability < threshold
            if class_info.probs[i] < cfg.min_prob:
                continue

            # don't get this until we need it, since it's expensive to calculate the first time
            highest_amplitude = self.get_highest_amplitude()

            # set prob = 0 if relative amplitude is too low
            amplitude = np.max(self.raw_spectrograms[i][low_index:high_index,:])
            relative_amplitude = amplitude / highest_amplitude
            if relative_amplitude < config.min_ratio:
                class_info.probs[i] = 0

    # The main config file has soundalike parameters for cases where a common species is mistaken for a rare one.
    # Here we handle cases where a common species is mistaken for a not-so-rare one. For example,
    # a fragment of a White-throated Sparrow song is sometimes mistaken for a Broad-winged Hawk.
    # If we're scanning BWHA and the current or previous label has a significant possibility of WTSP, skip this label.
    def check_soundalike(self, class_info):
        config = self.check_soundalike_config[class_info.code]
        if config.soundalike_code not in self.class_infos:
            return # must be using a subset of the full species list

        soundalike_info = self.class_infos[config.soundalike_code] # class_info for the soundalike species
        for i in range(len(class_info.probs)):
            # ignore if probability < threshold
            if class_info.probs[i] < cfg.min_prob:
                continue

            # set prob = 0 if current or previous soundalike prob >= min_prob
            if soundalike_info.probs[i] > config.min_prob or (i > 0 and soundalike_info.probs[i - 1] > config.min_prob):
                class_info.probs[i] = 0

    # Return the highest amplitude from the raw spectrograms;
    # Since they overlap, just check every 3rd one;
    # And skip the very lowest frequencies, which often contain loud noise
    def get_highest_amplitude(self):
        if self.highest_amplitude is None:
            self.highest_amplitude = 0
            for i in range(0, len(self.raw_spectrograms), 3):
                curr_max = np.max(self.raw_spectrograms[i][5:,:])
                self.highest_amplitude = max(self.highest_amplitude, curr_max)

        return self.highest_amplitude
