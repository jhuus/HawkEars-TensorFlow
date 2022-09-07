# Generate a png file for every spectrogram in the database for the given species.
# This is occasionally useful during development and tuning.

import argparse
import inspect
import os
import sys
import time
import matplotlib.pyplot as plt

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database
from core import plot
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-lnd', type=int, default=0, help='1 = low noise detector. Default = 0.')
parser.add_argument('-f', type=str, default='training', help='Database name.')
parser.add_argument('-c', type=int, default=0, help='1 = center the images.')
parser.add_argument('-g', type=int, default=0, help='1 = use gray scale.')
parser.add_argument('-n', type=int, default=0, help='If > 0, stop after this many images. Default = 0.')
parser.add_argument('-s', type=str, default='', help='Species name.')
parser.add_argument('-o', type=str, default='', help='Output directory.')
parser.add_argument('-p', type=str, default='', help='Only plot spectrograms if file name starts with this (case-insensitive).')
parser.add_argument('-w', type=int, default=0, help='1 = overwrite existing image files.')

args = parser.parse_args()

low_noise_detector = (args.lnd == 1)
db_name = args.f
species_name = args.s
prefix = args.p.lower()
num_to_plot = args.n
gray_scale = (args.g == 1)
center = (args.c == 1)
overwrite = (args.w == 1)
out_dir = args.o

if not os.path.exists(out_dir):
    print(f'creating directory {out_dir}')
    os.makedirs(out_dir)

db = database.Database(f'../data/{db_name}.db')

start_time = time.time()
results = db.get_spectrogram_details_by_name(species_name)
print(f'retrieved {len(results)} spectrograms from database')
num_plotted = 0
for result in results:
    filename, offset, spec, _ = result
    if len(prefix) > 0 and not filename.lower().startswith(prefix):
        continue

    base, ext = os.path.splitext(filename)
    spec_path = f'{out_dir}/{base}-{offset:.2f}.png'

    if overwrite or not os.path.exists(spec_path):
        print(f"Processing {spec_path}")
        spec = util.expand_spectrogram(spec, low_noise_detector=low_noise_detector)
        if center:
            spec = util.center_spec(spec)

        num_plotted += 1
        plot.plot_spec(spec, spec_path, low_noise_detector=low_noise_detector, gray_scale=gray_scale)

    if num_to_plot > 0 and num_plotted == num_to_plot:
        break

elapsed = time.time() - start_time
minutes = int(elapsed) // 60
seconds = int(elapsed) % 60
print(f'Elapsed time to plot {num_plotted} spectrograms = {minutes}m {seconds}s\n')

