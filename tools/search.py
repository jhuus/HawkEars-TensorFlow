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
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.spatial.distance
import skimage
from skimage import filters

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import config as cfg
from core import database
from core import util
from core import plot

'''
 Initialize
'''

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default='training', help='Database name (or upper case species code, or HNC).')
parser.add_argument('-g', type=int, default=0, help='1 = gray scale plots, 0 = colour. Default = 0.')
parser.add_argument('-m', type=float, default=0.6, help='Stop plotting when distance exceeds this. Default = 0.6.')
parser.add_argument('-n', type=int, default=60, help='Number of top matches to plot.')
parser.add_argument('-o', type=str, default='output', help='Output directory for plotting matches.')
parser.add_argument('-i', type=str, default='', help='Path to file containing spectrogram to search for.')
parser.add_argument('-s', type=str, default='', help='Species name.')
parser.add_argument('-t', type=float, default=0, help='Offset of spectrogram to search for.')
parser.add_argument('-x', type=str, default=None, help='If specified (e.g. "training"), skip spectrograms that exist in this database. Default = None.')
parser.add_argument('-z', type=int, default=0, help='1 means add blur, noise etc. to the spectrogram before searching. Default = 0.')

args = parser.parse_args()

ckpt_path = f'../data/{cfg.search_ckpt_name}'
db_name = args.f

target_path = args.i
target_offset = args.t
species_name = args.s
gray_scale = (args.g == 1)
max_dist = args.m
num_to_plot = args.n
out_dir = args.o
add_noise = (args.z == 1)
check_db_name = args.x

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

start_time = time.time()

# add random noise to the spectrogram (larger variances lead to more noise)
def _add_noise(spec, variance=0.003):
    spec = skimage.util.random_noise(spec, mode='gaussian', var=variance, clip=True)
    return spec

# blur the spectrogram (larger values of sigma lead to more blurring)
def _blur(spec, min_sigma=0.5, max_sigma=0.7):
    sigma = np.random.uniform(min_sigma, max_sigma)
    spec = skimage.filters.gaussian(spec, sigma=sigma, multichannel=False)
    return spec

# fade the spectrogram (smaller factors and larger min_vals lead to more fading);
# defaults don't have a big visible effect but do fade values a little, and it's
# important to preserve very faint spectrograms
def _fade(spec, min_factor=0.5, max_factor=0.7, min_val=0.07):
    factor = np.random.uniform(min_factor, max_factor)
    spec *= factor
    spec[spec < min_val] = 0 # clear values near zero
    spec *= 1/factor # rescale so max = 1
    spec = np.clip(spec, 0, 1) # just to be safe
    return spec

def _modify_spec(spec):
    # sequence of operations is important
    spec = _fade(spec)
    spec = _blur(spec)
    spec = _add_noise(spec)

    return spec

'''
 Get the spectrogram to search for, and plot it
'''

audio = audio.Audio(path_prefix='../')
signal, rate = audio.load(target_path)
specs = audio.get_spectrograms([target_offset])
if specs is None or len(specs) == 0:
    print(f'Failed to retrieve search spectrogram from offset {target_offset} in {target_path}')
    quit()

target_spec = specs[0].reshape((cfg.spec_height, cfg.spec_width, 1))

if add_noise:
    target_spec = _modify_spec(target_spec)

audio_file_name = os.path.basename(target_path)
_, ext = os.path.splitext(audio_file_name)
audio_file_name = audio_file_name[:-(len(ext))]
image_path = os.path.join(out_dir, f'0~{audio_file_name}~{target_offset:.2f}~0.0.png')
plot.plot_spec(target_spec, image_path, gray_scale=gray_scale)

'''
 Load the saved model
'''

print('loading saved model')
model = load_model(ckpt_path, compile=False)
encoder = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

'''
 Get spectrograms from the database
'''

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

results = db.get_spectrogram_by_subcat_name(species_name, include_embedding=True)

print(f'retrieved {len(results)} spectrograms to search')

max_len = 120000
if len(results) > max_len:
    # kludge: truncate results
    results = results[:max_len]

search_specs = np.zeros((len(results), cfg.spec_height, cfg.spec_width, 1), dtype=np.float32)

details = []
have_embeddings = False
for i, r in enumerate(results):
    spec = util.expand_spectrogram(r.value)
    spec = spec.reshape((cfg.spec_height, cfg.spec_width, 1))

    search_specs[i] = spec

    if r.embedding is None:
        details.append([0, r.filename, r.offset, spec])
    else:
        have_embeddings = True # assume they all have embeddings
        embedding = np.frombuffer(r.embedding, dtype=np.float16)
        details.append([embedding, r.filename, r.offset, spec])

check_spec_names = {}
if check_db_name is not None:
    check_db = database.Database(f'../data/{check_db_name}.db')
    results = check_db.get_spectrogram_by_subcat_name(species_name, include_embedding=True)
    for r in results:
        spec_name = f'{r.filename}-{r.offset:.2f}'
        check_spec_names[spec_name] = 1

'''
 Get the embeddings for the target spec and the database specs.
 Then compare them, sort the distances and plot the top ones.
'''

input = np.zeros((1, cfg.spec_height, cfg.spec_width, 1))
input[0] = target_spec
predictions = encoder.predict(input, verbose=0)
target_embedding = predictions[0]

if have_embeddings:
    for i in range(len(details)):
        # replace embedding with distance
        details[i][0] = scipy.spatial.distance.cosine(target_embedding, details[i][0])
else:
    predictions = encoder.predict(search_specs, verbose=0)
    for i in range(len(details)):
        details[i][0] = scipy.spatial.distance.cosine(target_embedding, predictions[i])

print('plotting results')
details = sorted(details, key=lambda value: value[0])
num_plotted = 0
spec_num = 0
for entry in details:
    if num_plotted == num_to_plot:
        break

    distance, filename, offset, spec = entry

    if distance > max_dist:
        break

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
