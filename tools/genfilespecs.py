# Generate a spectrogram png file per audio file in the given directory.
# This is occasionally useful during development and tuning.

import argparse
import inspect
import os
import sys
import time

import scipy

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import audio
from core import constants
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='', help='Directory containing the audio files. Subdirectory will be created for spectrograms.')
parser.add_argument('-n', type=int, default=9, help='Length of spectrograms in integer seconds. Default = 9.')
args = parser.parse_args()

root_dir = args.d
max_spec_len = args.n

out_dir = os.path.join(root_dir, 'filespecs')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

audio = audio.Audio(path_prefix='../')
for filename in os.listdir(root_dir):
    filepath = os.path.join(root_dir, filename)
    if util.is_audio_file(filepath):
        signal, rate = audio.load(filepath)
        if signal is None:
            continue
        
        spec_len = int(min(max_spec_len, signal.shape[0] / rate))
        width = int(spec_len * (constants.SPEC_WIDTH / constants.SEGMENT_LEN))
        if width < constants.SPEC_WIDTH:
            continue
        
        specs = audio.get_spectrograms([0], seconds=spec_len, shape=(constants.SPEC_HEIGHT, width), check_noise=False, exponent=0.4, row_factor=0)
        util.plot_spec(specs[0], os.path.join(out_dir, f'{filename}.png'))
       