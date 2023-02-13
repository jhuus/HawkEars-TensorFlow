# Species-specific handling during analysis / inference.
# To disable or add a handler, update self.handlers in the constructor below.

from types import SimpleNamespace

import numpy as np

from label import Label
from core import config as cfg

class Species_Handlers:
    def __init__(self, min_adj_prob, class_infos, offsets, merge_labels, labels, raw_spectrograms):
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

        self.min_adj_prob = min_adj_prob
        self.class_infos = {}
        for class_info in class_infos:
            self.class_infos[class_info.code] = class_info

        self.offsets = offsets
        self.merge_labels = merge_labels
        self.labels = labels
        self.raw_spectrograms = raw_spectrograms
        self.highest_amplitude = None

    # Handle cases where a faint vocalization is often mistaken for another species.
    # For example, distant songs of American Robin and similar-sounding species are sometimes mistaken for Pine Grosbeak.
    # So only create a Pine Grosbeak label if there is a relatively loud sound in the appropriate band of the spectrogram.
    def check_amplitude(self, name, class_info):
        config = self.check_amplitude_config[class_info.code]
        highest_amplitude = self.get_highest_amplitude()
        low_index = int(config.low_freq * cfg.spec_height)   # bottom of frequency range
        high_index = int(config.high_freq * cfg.spec_height) # top of frequency range

        prev_label = None
        for i, prob in enumerate(class_info.probs):
            # skip if probability < threshold
            if prob < cfg.min_prob:
                continue

            end_time = self.offsets[i] + cfg.segment_len
            if cfg.check_adjacent and i not in [0, len(class_info.probs) - 1]:
                # skip if adjacent label probabilities are too low
                if class_info.probs[i - 1] < self.min_adj_prob and class_info.probs[i + 1] < self.min_adj_prob:
                    continue

            # skip if relative amplitude is too low
            amplitude = np.max(self.raw_spectrograms[i][low_index:high_index,:])
            relative_amplitude = amplitude / highest_amplitude
            if relative_amplitude < config.min_ratio:
                continue

            if self.merge_labels and prev_label != None and prev_label.end_time >= self.offsets[i]:
                # extend the previous label's end time (i.e. merge)
                prev_label.end_time = end_time
                prev_label.probability = max(prob, prev_label.probability)
            else:
                label = Label(name, prob, self.offsets[i], end_time)
                self.labels.append(label)
                prev_label = label

    # The main config file has soundalike parameters for cases where a common species is mistaken for a rare one.
    # Here we handle cases where a common species is mistaken for a not-so-rare one. For example,
    # a fragment of a White-throated Sparrow song is sometimes mistaken for a Broad-winged Hawk.
    # If we're scanning BWHA and the current or previous label has a significant possibility of WTSP, skip this label.
    def check_soundalike(self, name, class_info):
        config = self.check_soundalike_config[class_info.code]
        if config.soundalike_code in self.class_infos:
            soundalike_info = self.class_infos[config.soundalike_code]
        else:
            soundalike_info = None

        prev_label = None
        for i, prob in enumerate(class_info.probs):
            # skip if probability < threshold
            if prob < cfg.min_prob:
                continue

            end_time = self.offsets[i] + cfg.segment_len
            if cfg.check_adjacent and i not in [0, len(class_info.probs) - 1]:
                # skip if adjacent label probabilities are too low
                if class_info.probs[i - 1] < self.min_adj_prob and class_info.probs[i + 1] < self.min_adj_prob:
                    continue

            if soundalike_info is not None:
                # skip if current or previous soundalike prob >= min_prob
                if soundalike_info.probs[i] > config.min_prob or (i > 0 and soundalike_info.probs[i - 1] > config.min_prob):
                    continue

            if self.merge_labels and prev_label != None and prev_label.end_time >= self.offsets[i]:
                # extend the previous label's end time (i.e. merge)
                prev_label.end_time = end_time
                prev_label.probability = max(prob, prev_label.probability)
            else:
                label = Label(name, prob, self.offsets[i], end_time)
                self.labels.append(label)
                prev_label = label

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
