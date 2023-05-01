# Delete spectrograms from the database. Input is either a text file or a directory of images.
# In a text file, each line has the format "filename-offset", e.g. "XC10503-27.0" to identify the spectrogram.
# In a directory of images, each image name must have that format.

import argparse
import inspect
import os
import sys

from pathlib import Path

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database
from core import util

def fatal_error(msg):
    print(msg)
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='training', help='Database name. Default = training')
parser.add_argument('-i', type=str, default=None, help='Input text file or directory name. No default.')
parser.add_argument('-s', type=str, default='', help='Species name')
args = parser.parse_args()

db = database.Database(filename=f'../data/{args.f}.db')
input = args.i

# need to get recording by species in case same file name is used for different species
species_name = args.s
recording_dict = {}
results = db.get_recording_by_subcat_name(species_name)
for r in results:
    tokens = r.filename.split('.')
    recording_dict[tokens[0]] = r.id

if Path(input).is_file():
    spec_names = util.get_file_lines(input)
else:
    temp = os.listdir(input)
    spec_names = []
    for file_name in temp:
        if os.path.isfile(os.path.join(input, file_name)):
            base, ext = os.path.splitext(file_name)
            if ext == '.png':
                spec_names.append(base)

for spec_name in spec_names:
    index = spec_name.rfind('-')
    if index == -1:
        fatal_error(f'invalid spectrogram name: {spec_name}')

    recording_name = spec_name[:index]
    offset = float(spec_name[index+1:])

    if recording_name in recording_dict.keys():
        recording_id = recording_dict[recording_name]
    else:
        fatal_error(f'recording not found: {recording_name}')

    result = db.get_spectrogram_by_recid_and_offset(recording_id, offset)
    if result is None:
        print(f'spectrogram not found: {recording_name}-{offset}')
    else:
        print(f'deleting spectrogram ID {result.id}')
        db.delete_spectrogram('ID', result.id)
