# Create a new database containing spectrograms and their embeddings for a single species,
# or add new spectrograms and their embeddings to an existing database.
# This is used to quickly search a large collection of recordings for matching spectrograms.

import argparse
import inspect
from multiprocessing import Process
import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.spatial.distance

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import database
from core import config as cfg
from core import util
from core import plot

class Main:
    def __init__(self, ckpt_path, source_dir, data_dir, check_peaks):
        self.source_dir = source_dir
        self.data_dir = data_dir
        self.check_peaks = check_peaks
        self.ckpt_path = ckpt_path

    def run(self, file_list):
        # initialize
        audio_obj = audio.Audio(path_prefix='../')

        db_path = os.path.join(self.data_dir, f'{code}.db')
        db = database.Database(filename=db_path)

        model = keras.models.load_model(self.ckpt_path, compile=False)
        encoder = keras.models.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

        # get list of existing recordings so we can skip those and only add the new ones
        source_name = 'dummy'
        results = db.get_source('Name', source_name)
        if len(results) == 0:
            source_id = db.insert_source(source_name)
        else:
            source_id = results[0].id

        category_name = 'bird'
        results = db.get_category('Name', category_name)
        if len(results) == 0:
            category_id = db.insert_category(category_name)
        else:
            category_id = results[0].id

        results = db.get_subcategory_by_catid_and_subcat_name(category_id, species_name)
        if len(results) == 0:
            subcategory_id = db.insert_subcategory(category_id, species_name, code=code)
        else:
            subcategory_id = results[0].id

        existing_recordings = {}
        results = db.get_recording_by_subcat_name(species_name)
        for r in results:
            existing_recordings[r.filename] = 1

        # get spectrograms from the file system
        details = []
        for file_path in file_list:
            filename = os.path.basename(file_path)
            if filename in existing_recordings:
                # skip recordings that are already in the database
                continue

            print(f'processing {file_path}')
            signal, rate = audio_obj.load(file_path)
            if signal is None:
                continue

            seconds = int(len(signal) / rate)
            if self.check_peaks:
                offsets = audio_obj.find_sounds()
            else:
                end_seconds = max(1, (len(signal) / rate) - cfg.segment_len)
                offsets = np.arange(0, end_seconds, 1.5).tolist()

            specs = audio_obj.get_spectrograms(offsets)

            for i, spec in enumerate(specs):
                spec = spec.reshape((cfg.spec_height, cfg.spec_width, 1))
                filename = os.path.basename(file_path)
                details.append([filename, offsets[i], spec, seconds, 0])

        if len(details) == 0:
            return

        # get embeddings
        print('getting embeddings')
        specs = np.zeros((len(details), cfg.spec_height, cfg.spec_width, 1))
        for i in range(len(specs)):
            specs[i] = details[i][2]

        predictions = encoder.predict(specs, verbose=0)
        for i in range(len(predictions)):
            details[i][4] = predictions[i]

        # insert recording and spectrogram records
        print('inserting database records')
        file_hash = {}
        for detail in details:
            filename, offset, spec, seconds, embedding = detail
            if filename in file_hash.keys():
                recording_id = file_hash[filename]
            else:
                recording_id = db.insert_recording(source_id, subcategory_id, filename, seconds)
                file_hash[filename] = recording_id

            compressed_spec = util.compress_spectrogram(spec)
            db.insert_spectrogram(recording_id, compressed_spec, offset, embedding=embedding)

if __name__ == '__main__':
    # rather than specifying r1 and r2 arguments every time, you can save them
    # in environment variables SOURCE_DIR and DATA_DIR respectively
    source_dir_env = os.environ.get('SOURCE_DIR')
    data_dir_env = os.environ.get('DATA_DIR')

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r1', type=str, default=source_dir_env, help='Root directory for files to scan.')
    parser.add_argument('-r2', type=str, default=data_dir_env, help='Root directory to write database under.')
    parser.add_argument('-s', type=str, default='', help='Species name.')
    parser.add_argument('-c', type=str, default='', help='Species code.')
    parser.add_argument('-p', type=int, default=0, help='0 means extract at 1.5 second offsets, 1 means extract at peaks. Default = 0.')

    args = parser.parse_args()

    ckpt_path = f'../data/{cfg.search_ckpt_name}'
    species_name = args.s
    code = args.c
    check_peaks = args.p == 1

    if len(species_name) == 0:
        print('Error: species name must be specified')
        quit()

    if len(code) == 0:
        print('Error: species code must be specified')
        quit()

    source_dir = os.path.join(args.r1, code)
    data_dir = os.path.join(args.r2, code)

    if not os.path.exists(source_dir):
        print(f'Error: path not found: "{source_dir}"')
        quit()

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    start_time = time.time()

    # do a batch at a time to avoid running out of GPU memory
    file_list = util.get_audio_files(source_dir)
    print(f'{len(file_list)} audio files found in {source_dir}')
    start_idx = 0
    while start_idx < len(file_list):
        end_idx = min(start_idx + cfg.analyze_group_size, len(file_list))

        print(f'Processing files {start_idx} to {end_idx - 1}')
        main = Main(ckpt_path, source_dir, data_dir, check_peaks)
        p = Process(target=main.run, args=((file_list[start_idx:end_idx],)))
        p.start()
        p.join()
        start_idx += cfg.analyze_group_size

    elapsed = time.time() - start_time
    print(f'elapsed seconds = {elapsed:.3f}')
