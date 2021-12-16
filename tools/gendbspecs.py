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

from core import audio
from core import constants
from core import database
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='training', help='Database name.')
parser.add_argument('-s', type=str, default='', help='Species name.')
parser.add_argument('-o', type=str, default='', help='Output directory.')
parser.add_argument('-p', type=str, default='', help='Only plot spectrograms if file name starts with this (case-insensitive).')
parser.add_argument('-n', type=int, default=constants.SPEC_HEIGHT, help='Number of rows to plot. Default is all.')

args = parser.parse_args()

db_name = args.f
species_name = args.s
prefix = args.p.lower()
num_rows = args.n
out_dir = args.o
if not os.path.exists(out_dir):
    print(f'creating directory {out_dir}')
    os.makedirs(out_dir)

db = database.Database(f'../data/{db_name}.db')
audio = audio.Audio(path_prefix='../')

start_time = time.time()
results = db.get_spectrogram_details_by_name(species_name)
num_plotted = 0
for result in results:
    filename, offset, spec = result
    if len(prefix) > 0 and not filename.lower().startswith(prefix):
        continue
    
    base, ext = os.path.splitext(filename)
    spec = util.expand_spectrogram(spec)
    num_plotted += 1
    #util.plot_spec(spec[:num_rows,:], f'{out_dir}/{base}-{offset:.2f}.png')
    spec = spec.reshape((constants.SPEC_HEIGHT, constants.SPEC_WIDTH))    
    plt.clf() # clear any existing plot data
    plt.pcolormesh(spec[:num_rows,:], shading='gouraud')
    plt.savefig(f'{out_dir}/{base}-{offset:.2f}.png')
    
elapsed = time.time() - start_time
minutes = int(elapsed) // 60
seconds = int(elapsed) % 60
print(f'Elapsed time to plot {num_plotted} spectrograms = {minutes}m {seconds}s\n')
    
