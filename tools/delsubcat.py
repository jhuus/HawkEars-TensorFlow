# Delete all spectrograms and recordings for a subcategory (i.e. class or species) from a database.

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import database

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='training', help='Database name. Default = training')
parser.add_argument('-s', type=str, default='', help='Species name')
args = parser.parse_args()

db_name = args.f
species_name = args.s

database = database.Database(f'../data/{db_name}.db')
database.delete_spectrogram(species_name)
database.delete_recording(species_name)
