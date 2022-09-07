# List (and optionally move) audio files that are in a directory but not in the specified database(s),
# given a species name.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database
from core import util

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f1', type=str, default='training', help='Database name #1.')
parser.add_argument('-f2', type=str, default='validation', help='Database name #2.')
parser.add_argument('-s', type=str, default='', help='Species name.')
parser.add_argument('-i', type=str, default='', help='Input directory.')
parser.add_argument('-o', type=str, default='', help='If specified, move the unused files to this directory.')

args = parser.parse_args()

db_names = [args.f1]
if len(args.f2) > 0:
    db_names.append(args.f2)

species_name = args.s
input_dir = args.i
output_dir = args.o

if len(output_dir) > 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create a list of file names in the input directory (without the '.mp3' extension)
raw_list = util.get_audio_files(input_dir)
base_to_file_dict = {}
input_dir_list = []
for file_path in raw_list:
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    input_dir_list.append(base_name)
    base_to_file_dict[base_name] = file_name

# create a list of file names in the database(s) (also without the '.mp3' extension)
db_dict = {}
for db_name in db_names:
    db = database.Database(f'../data/{db_name}.db')
    results = db.get_recordings_by_subcategory_name(species_name)
    for result in results:
        _, file_name, _ = result
        base_name, ext = os.path.splitext(file_name)
        db_dict[base_name] = 1

# list files that are in the input directory but not the database(s)
sep = os.path.sep
for base_name in input_dir_list:
    if base_name not in db_dict.keys():
        if len(output_dir) > 0:
            cmd = f'mv "{input_dir}{sep}{base_to_file_dict[base_name]}" "{output_dir}{sep}{base_to_file_dict[base_name]}"'
            print(cmd)
            os.system(cmd)
        else:
            print(base_name)
