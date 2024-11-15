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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1 = no info, 2 = no warnings, 3 = no errors

from core import audio
from core import config as cfg
from core import plot
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='', help='Directory containing the audio files. Subdirectory will be created for spectrograms.')
parser.add_argument('-l', type=int, default=0, help='If 1, use low_band audio settings (for Ruffed Grouse drumming identifier)')
parser.add_argument('-n', type=int, default=9, help='Length of spectrograms in integer seconds. Default = 9.')
args = parser.parse_args()

root_dir = args.d
low_band = args.l == 1
max_spec_len = args.n

out_dir = os.path.join(root_dir, 'filespecs')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

audio_obj = None
for file_name in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file_name)
    if util.is_audio_file(file_path):
        image_path = os.path.join(out_dir, f'{file_name}.png')
        if os.path.exists(image_path):
            continue

        if audio_obj is None:
            audio_obj = audio.Audio(path_prefix='../')

        print(f'processing {file_path}')
        signal, rate = audio_obj.load(file_path)
        if signal is None:
            continue

        spec_len = int(min(max_spec_len, signal.shape[0] / rate))
        width = int(spec_len * (cfg.spec_width / cfg.segment_len))
        if width < cfg.spec_width:
            continue

        specs = audio_obj.get_spectrograms([0], segment_len=spec_len, low_band=low_band)
        plot.plot_spec(specs[0], image_path)
