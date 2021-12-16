# Shared constants

SEGMENT_LEN = 3     # duration per spectrogram in seconds
SPEC_HEIGHT = 80    # normal spectrogram height
SPEC_WIDTH = 384    # spectrogram width
BINARY_SPEC_HEIGHT = 20 # binary classifier spectrogram height

IGNORE_FILE = 'data/ignore.txt' # classes listed in this file are ignored in analysis
CLASSES_FILE = 'data/classes.txt'     # list of classes used in training and analysis
CKPT_PATH = 'data/ckpt'               # where to save/load model checkpoint
BINARY_CKPT_PATH = 'data/binary_classifier_ckpt' # model checkpoint for binary classifier
