# Update the embeddings of all spectrograms in the database.

import argparse
import os
import inspect
from multiprocessing import Process
import sys
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from tensorflow import keras

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import config as cfg
from core import database
from core import util

BATCH_SIZE = 2048 # process this many spectrograms at a time

def get_database(db_name):
    # Upper-case 4-letter db names are assumed to refer to $(DATA_DIR)/{code}/{code}.db;
    # e.g. "AMRO" refers to $(DATA_DIR)/AMRO/AMRO.db.
    # Otherwise we assume it refers to ../data/{db name}.db.
    data_dir = os.environ.get('DATA_DIR')
    if data_dir is not None and len(db_name) == 4 and db_name.isupper():
        # db name is a species code (or dummy species code in some cases)
        db = database.Database(f'{data_dir}/{db_name}/{db_name}.db')
    else:
        db = database.Database(f'../data/{db_name}.db')

    return db

class Main:
    def __init__(self, db_name, species_name):
        self.db_name = db_name
        self.species_name = species_name

    def run(self, specs):
        self.db = get_database(db_name)
        ckpt_path = f'../data/{cfg.search_ckpt_name}'
        self.model = keras.models.load_model(ckpt_path, compile=False)
        self.encoder = keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer('avg_pool').output)

        spec_array = np.zeros((len(specs), cfg.spec_height, cfg.spec_width, 1))
        ids = []
        for i, spec in enumerate(specs):
            ids.append(spec.id)
            spec = util.expand_spectrogram(spec.value)
            spec = spec.reshape((cfg.spec_height, cfg.spec_width, 1))
            spec_array[i] = spec

        predictions = self.encoder.predict(spec_array, verbose=0)
        for i in range(len(predictions)):
            self.db.update_spectrogram(ids[i], 'Embedding', predictions[i])

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='training', help='Database name. Default = "training"')
    parser.add_argument('-s', type=str, default='', help='Optional species name. Default = blank, so update all species.')
    args = parser.parse_args()

    db_name = args.f
    species_name = args.s

    start_time = time.time()

    # do a batch at a time to avoid running out of GPU memory
    db = get_database(db_name)
    if len(species_name) == 0:
        # get all subcategories
        results = db.get_subcategory()
    else:
        # get the requested subcategory
        results = db.get_subcategory('Name', species_name)

    if len(results) == 0:
        print(f'Failed to retrieve species information from database.')
        quit()

    for result in results:
        print(f'Processing {result.name}')
        specs = db.get_spectrogram_by_subcat_name(result.name, include_ignored=True)
        start_idx = 0

        while start_idx < len(specs):
            end_idx = min(start_idx + BATCH_SIZE, len(specs))
            print(f'Processing spectrograms {start_idx} to {end_idx - 1}')

            main = Main(db_name, result.name)
            p = Process(target=main.run, args=((specs[start_idx:end_idx],)))
            p.start()
            p.join()
            start_idx += BATCH_SIZE

    elapsed = time.time() - start_time
    print(f'elapsed seconds = {elapsed:.3f}')
