# generate the curve used to compensate for mel spectrogram amplitude distortion

import argparse
import os
import inspect
import sys

import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import config as cfg

DEFAULT_COUNT = 2000
DEFAULT_DIR = 'mel_output'

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=0, help='Mode 0 = generate code for curve, 1 = output white noise amplitude per frequency without adjustment, 2 = output csv with adjustment. Default = 0.')
parser.add_argument('-n', type=int, default=DEFAULT_COUNT, help=f'Number of white noise spectrograms to generate. Higher counts generate a more precise curve. Default = {DEFAULT_COUNT}.')
parser.add_argument('-o', type=str, default=DEFAULT_DIR, help=f'Output directory path. Default = "{DEFAULT_DIR}".')

args = parser.parse_args()
mode = args.m
count = args.n
output_dir = args.o

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

audio = audio.Audio(path_prefix='../')
if mode == 2:
    cfg.mel_amplitude_adjustment = True
else:
    cfg.mel_amplitude_adjustment = False

# get average amplitude per frequency for each of count white noise spectrograms
amplitudes = np.zeros((cfg.spec_height, count))
for i in range(count):
    spec = audio.white_noise()
    for j in range(cfg.spec_height):
        amplitudes[j, i] = np.average(spec[j])

if mode == 0:
    # output as text that can be pasted into audio.py
    txt_path = os.path.join(output_dir, 'code.txt')
    with open(txt_path, 'w') as txt_file:
        txt_file.write('adjust_mel_amplitude = [\n')
        for i in range(int(cfg.spec_height / 16)):
            str = '    '
            for j in range(16):
                str += f'{np.average(amplitudes[i * 16 + j]):.6f},'
            txt_file.write(f'{str}\n')
        txt_file.write(']\n')

    print(f'See output in {txt_path}')
else:
    # print average amplitude per frequency across all examples
    csv_path = os.path.join(output_dir, 'amplitudes.csv')
    with open(csv_path, 'w') as csv_file:
        for i in range(cfg.spec_height):
            csv_file.write(f'{np.average(amplitudes[i])}\n')

    print(f'See output in {csv_path}')
