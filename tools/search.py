# Search a database for spectrograms similar to a given one.
# Main inputs are a path and offset to specify the search spectrogram,
# and a species name to search in the database.

import argparse
import inspect
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1 = no info, 2 = no warnings, 3 = no errors

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.spatial.distance

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import config as cfg
from core import database
from core import util
from core import plot

class SpecInfo:
    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='training', help='Database name (or upper case species code, or HNC).')
parser.add_argument('-g', type=int, default=0, help='1 = gray scale plots, 0 = colour. Default = 0.')
parser.add_argument('-m', type=float, default=0.6, help='Stop plotting when distance exceeds this. Default = 0.6.')
parser.add_argument('-n', type=int, default=60, help='Number of top matches to plot.')
parser.add_argument('-o', type=str, default='output', help='Output directory for plotting matches.')
parser.add_argument('-i', type=str, default='', help='Path to file containing spectrogram to search for.')
parser.add_argument('-s', type=str, default=None, help='Species name to search for.')
parser.add_argument('-s2', type=str, default=None, help='Species name to use in target DB if -x is specified. If this is omitted, default to -s option.')
parser.add_argument('-t', type=float, default=0, help='Offset of spectrogram to search for.')
parser.add_argument('-x', type=str, default=None, help='If specified (e.g. "training"), skip spectrograms that exist in this database. Default = None.')

args = parser.parse_args()

ckpt_path = f'../data/{cfg.search_ckpt_name}'
db_name = args.f

target_path = args.i
target_offset = args.t
species_name = args.s
skip_species_name = args.s2
gray_scale = (args.g == 1)
max_dist = args.m
num_to_plot = args.n
out_dir = args.o
check_db_name = args.x

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

start_time = time.time()

# get the spectrogram to search for, and plot it
audio = audio.Audio(path_prefix='../')
signal, rate = audio.load(target_path)
specs = audio.get_spectrograms([target_offset])
if specs is None or len(specs) == 0:
    print(f'Failed to retrieve search spectrogram from offset {target_offset} in {target_path}')
    quit()

target_spec = specs[0].reshape((cfg.spec_height, cfg.spec_width, 1))

audio_file_name = os.path.basename(target_path)
_, ext = os.path.splitext(audio_file_name)
audio_file_name = audio_file_name[:-(len(ext))]
image_path = os.path.join(out_dir, f'0~{audio_file_name}~{target_offset:.2f}~0.0.png')
plot.plot_spec(target_spec, image_path, gray_scale=gray_scale)

# get spectrograms from the database
print('opening database')

# Upper-case 4-letter db names are assumed to refer to $(DATA_DIR)/{code}/{code}.db;
# e.g. "AMRO" refers to $(DATA_DIR)/AMRO/AMRO.db.
# Otherwise we assume it refers to ../data/{db name}.db.
data_dir = os.environ.get('DATA_DIR')
if data_dir is not None and len(db_name) == 4 and db_name.isupper():
    # db name is a species code (or dummy species code in some cases)
    db = database.Database(f'{data_dir}/{db_name}/{db_name}.db')
else:
    db = database.Database(f'../data/{db_name}.db')

# get recordings and create dict from ID to filename
recording_dict = {}
if species_name is None:
    results = db.get_recording()
else:
    results = db.get_recording_by_subcat_name(species_name)

for r in results:
    recording_dict[r.id] = r.filename

# get embeddings only, since getting spectrograms here might use too much memory
if species_name is None:
    results = db.get_spectrogram_embeddings()
else:
    results = db.get_spectrogram_embeddings_by_subcat_name(species_name)

print(f'retrieved {len(results)} spectrograms to search')

spec_infos = []
for i, r in enumerate(results):
    if r.embedding is None:
        print('Error: not all spectrograms have embeddings')
        quit()
    else:
        embedding = np.frombuffer(r.embedding, dtype=np.float16)
        spec_infos.append(SpecInfo(r.id, embedding))

check_spec_names = {}
if check_db_name is not None:
    check_db = database.Database(f'../data/{check_db_name}.db')
    use_name = skip_species_name if skip_species_name is not None else species_name
    results = check_db.get_spectrogram_by_subcat_name(use_name, include_embedding=True)
    for r in results:
        spec_name = f'{r.filename}-{r.offset:.2f}'
        check_spec_names[spec_name] = 1

# load the saved model, i.e. the search checkpoint
print('loading saved model')
model = keras.models.load_model(ckpt_path, compile=False)
encoder = keras.models.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

# get the embedding for the target spectrogram
input = np.zeros((1, cfg.spec_height, cfg.spec_width, 1))
input[0] = target_spec
predictions = encoder.predict(input, verbose=0)
target_embedding = predictions[0]

# compare embeddings and save the distances
print('comparing embeddings')
for i in range(len(spec_infos)):
    spec_infos[i].distance = scipy.spatial.distance.cosine(target_embedding, spec_infos[i].embedding)

# sort by distance and plot the results
print('sorting results')
spec_infos = sorted(spec_infos, key=lambda value: value.distance)

print('plotting results')
num_plotted = 0
spec_num = 0
for spec_info in spec_infos:
    if num_plotted == num_to_plot or spec_info.distance > max_dist:
        break

    results = db.get_spectrogram('ID', spec_info.id, include_ignored=True)
    if len(results) != 1:
        print(f'Error: unable to retrieve spectrogram {spec_info.id}')

    filename = recording_dict[results[0].recording_id]
    offset = results[0].offset
    distance = spec_info.distance
    spec = util.expand_spectrogram(results[0].value)
    spec = spec.reshape((cfg.spec_height, cfg.spec_width, 1))

    spec_name = f'{filename}-{offset:.2f}'
    if spec_name in check_spec_names:
        continue

    spec_num += 1
    base, ext = os.path.splitext(filename)
    spec_path = os.path.join(out_dir, f'{spec_num}~{base}-{offset:.2f}~{distance:.3f}.png')

    if not os.path.exists(spec_path):
        plot.plot_spec(spec, spec_path, gray_scale=gray_scale)
        num_plotted += 1

elapsed = time.time() - start_time
print(f'elapsed seconds = {elapsed:.3f}')
