# Delete spectrogram images from the filesystem. The file dir/specs.txt lists the spectrograms
# to keep, so delete any not found there, where each line has the format "filename-offset", 
# e.g. "XC10503-27.0". Images are assumed to be in dir/spectrograms.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import util

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='', help='Path for this species.')
args = parser.parse_args()

root_dir = args.d
spec_dir = os.path.join(root_dir, 'spectrograms')

spec_names = util.get_file_lines(os.path.join(root_dir, 'specs.txt'))
spec_dict = {}
for name in spec_names:
    spec_dict[name] = 1

for file_name in os.listdir(spec_dir):
    file_path = os.path.join(spec_dir, file_name) 
    if os.path.isfile(file_path):
        base, ext = os.path.splitext(file_name)
        if base not in spec_dict:
            os.remove(file_path)
