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
parser.add_argument('-i', type=str, default='discards.txt', help='Input text file or directory name. Default = discards.txt')
parser.add_argument('-s', type=str, default='', help='Species name')
args = parser.parse_args()

db = database.Database(filename=f'../data/{args.f}.db')
input = args.i

# need to get recording by species in case same file name is used for different species
species_name = args.s
recording_dict = {}
results = db.get_recordings_by_subcategory_name(species_name)
for result in results:
    recording_id, file_name, _ = result
    recording_dict[file_name[:-4]] = recording_id

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

    result = db.get_spectrogram(recording_id, offset)
    if result is None:
        print(f'spectrogram not found: {recording_name}-{offset}')
    else:
        spec_id, value = result
        print(f'deleting spectrogram ID {spec_id}')
        db.delete_spectrogram_by_id(spec_id)
