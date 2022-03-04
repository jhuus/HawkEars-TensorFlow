# Shared constants

SEGMENT_LEN = 3         # duration per spectrogram in seconds
FMIN = 30               # min frequency
FMAX = 12500            # max frequency
SAMPLING_RATE = 44100
WIN_LEN = 2048          # FFT window length

BINARY_SPEC_HEIGHT = 32 # binary classifier spectrogram height
SPEC_HEIGHT = 128       # normal spectrogram height
SPEC_WIDTH = 384        # spectrogram width

IGNORE_FILE = 'data/ignore.txt'     # classes listed in this file are ignored in analysis
CLASSES_FILE = 'data/classes.txt'   # list of classes used in training and analysis
CKPT_PATH = 'data/ckpt'             # where to save/load model checkpoint
BINARY_CKPT_PATH = 'data/binary_classifier_ckpt' # model checkpoint for binary classifier
