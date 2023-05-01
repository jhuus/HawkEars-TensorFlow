# This script extracts spectrograms from audio files and inserts them into the database.
# The list of spectrograms comes from either a text file or a directory of images.
# In a text file, each line has the format "filename-offset", e.g. "XC10503-27.0" to identify the spectrogram.
# In a directory of images, each image name must have that format.

import argparse
import inspect
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1 = no info, 2 = no warnings, 3 = no errors

from core import audio
from core import config as cfg
from core import database
from core import util

class Spectrogram:
    def __init__(self, spec, name, recording_id, offset):
        self.spec = spec
        self.name = name
        self.recording_id = recording_id
        self.offset = offset

class Main:
    def __init__(self, source, category, code, root, dbname, input_path, subcategory, target_dir, target_dir2, use_low_band):
        self.db = database.Database(filename=f'../data/{dbname}.db')
        self.audio = audio.Audio(path_prefix='../')

        self.source = source
        self.category = category
        self.code = code
        self.root = root
        self.dbname = dbname
        self.input_path = input_path
        self.subcategory = subcategory
        self.target_dir = target_dir
        self.target_dir2 = target_dir2
        self.use_low_band = use_low_band

    def _get_offsets(self, filename):
        base_name = Path(filename).stem
        if base_name in self.offsets_dict.keys():
            return self.offsets_dict[base_name]
        else:
            return []

    def _get_source_id(self, filename):
        if self.source is not None:
            return self.source_id

        source_name = util.get_source_name(filename)
        if source_name in self.source_ids:
            return self.source_ids[source_name]
        else:
            source_id = self.db.insert_source(source_name)
            self.source_ids[source_name] = source_id
            return source_id

    # get a 3-second spectrogram at each of the given offsets of the specified file
    def _get_spectrograms(self, path, offsets):
        specs = []
        signal, rate = self.audio.load(path)
        seconds = len(signal) / rate

        filename = os.path.basename(path)
        index = filename.rfind('.')
        prefix = filename[:index]
        source_id = self._get_source_id(prefix)

        results = self.db.get_recording_by_src_subcat_file(source_id, self.subcategory_id, filename)
        if len(results) == 0:
            recording_id = self.db.insert_recording(source_id, self.subcategory_id, filename, seconds)
        else:
            recording_id = results[0].id

        spec_exponent = .8 # use a slightly higher value for extracting training data than for analysis, so training data is a bit cleaner
        raw_specs = self.audio.get_spectrograms(offsets, spec_exponent=spec_exponent, low_band=self.use_low_band)
        for i in range(len(offsets)):
            spec = raw_specs[i]
            specs.append(Spectrogram(spec, f'{prefix}-{offsets[i]:.2f}', recording_id, offsets[i]))

        return specs, raw_specs

    # insert spectrograms into the database
    def _import_spectrograms(self):
        print(f'import spectrograms')
        inserted = 0
        audio_files = util.get_audio_files(self.root)
        specs = []
        raw_specs = []
        print(f'Found {len(audio_files)} audio files in {self.root}')
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            print(f'processing {filename}')

            offsets = self._get_offsets(filename)
            if len(offsets) == 0:
                continue

            curr_specs, curr_raw_specs = self._get_spectrograms(audio_file, offsets)
            specs.extend(curr_specs)
            raw_specs.extend(curr_raw_specs)

        for spec in specs:
            if not spec.spec is None:
                # convert to bytes, zip it and insert in database
                compressed = util.compress_spectrogram(spec.spec)
                self.db.insert_spectrogram(spec.recording_id, compressed, spec.offset)
                inserted += 1

        print(f'inserted {inserted} spectrograms')

    def _init_database(self):
        if self.source is not None:
            results = self.db.get_source('Name', self.source)
            if len(results) == 0:
                self.source_id = self.db.insert_source(self.source)
            else:
                self.source_id = results[0].id

        results = self.db.get_category('Name', self.category)
        if len(results) == 0:
            self.category_id = self.db.insert_category(self.category)
        else:
            self.category_id = results[0].id

        results = self.db.get_subcategory_by_catid_and_subcat_name(self.category_id, self.subcategory)
        if len(results) == 0:
            self.subcategory_id = self.db.insert_subcategory(self.category_id, self.subcategory, code=self.code)
        else:
            self.subcategory_id = results[0].id

    # read file containing spec info and update self.offsets_dict
    def _process_specs_file(self, specs_file_path):
        for line in util.get_file_lines(specs_file_path):
            try:
                line = line.strip()
                if len(line) > 0:
                    pos = line.rindex('-')
                    filename = line[:pos]
                    offset = float(line[pos+1:])

                    if filename not in self.offsets_dict.keys():
                        self.offsets_dict[filename] = []

                    self.offsets_dict[filename].append(offset)
            except:
                continue

    def _process_spec_name(self, name):
        if '-' not in name:
            print(f'Skipping {name} (bad name)')
            return

        pos = name.rindex('-')
        filename = name[:pos]
        offset = float(name[pos+1:])

        if filename not in self.offsets_dict.keys():
            self.offsets_dict[filename] = []

        self.offsets_dict[filename].append(offset)

    # after doing everything else, it is sometimes useful to copy the used audio files to a target directory
    def _copy_files(self):
        target_dir = None
        if len(self.target_dir2) > 0:
            target_dir = self.target_dir2
        elif len(self.target_dir) > 0 and len(self.code) > 0:
            target_dir = os.path.join(self.target_dir, self.code)

        if target_dir is not None:
            for source_path in util.get_audio_files(self.root):
                filename = os.path.basename(source_path)
                base, ext = os.path.splitext(filename)

                if base in self.offsets_dict.keys():
                    target_path = os.path.join(target_dir, filename)
                    source_path = os.path.join(self.root, filename)
                    if not os.path.exists(target_path):
                        print(f'copy {filename} to {target_path}')
                        shutil.copyfile(source_path, target_path)

    def run(self):
        self._init_database()

        results = self.db.get_source()
        self.source_ids = {} # source ID per name
        for r in results:
            self.source_ids[r.name] = r.id

        # get label/specs info unless we're generating labels
        self.offsets_dict = {}
        if os.path.isfile(self.input_path):
            for line in util.get_file_lines(self.input_path):
                try:
                    line = line.strip()
                    if len(line) > 0:
                        self._process_spec_name(line)
                except:
                    continue
        else:
            # input path must point to a directory of images
            file_list = os.listdir(self.input_path)
            print(f'Found {len(file_list)} files in {self.input_path}')
            for file_name in file_list:
                base, ext = os.path.splitext(file_name)
                if ext == '.png':
                    self._process_spec_name(base)

        self._import_spectrograms()
        self._copy_files()
        self.db.close()

if __name__ == '__main__':
    source_dir_env = os.environ.get('SOURCE_DIR')
    data_dir_env = os.environ.get('DATA_DIR')

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, default=None, help='Source (e.g. "Xeno-Canto").')
    parser.add_argument('-b', type=str, default='bird', help='Category (e.g. bird).')
    parser.add_argument('-c', type=str, default='', help='Code (e.g. BAOR).')
    parser.add_argument('-d', type=str, default=None, help='Directory containing the audio files.')
    parser.add_argument('-f', type=str, default='training', help='Database name. Default = training')
    parser.add_argument('-i', type=str, default='', help='Input text file or image directory')
    parser.add_argument('-l', type=int, default=0, help='If 1, use low_band audio settings (for Ruffed Grouse drumming identifier)')
    parser.add_argument('-s', type=str, default='', help='Subcategory (e.g. "Baltimore Oriole").')
    parser.add_argument('-t', type=str, default=data_dir_env, help='Copy used audio files to a "bird code" directory under this, if specified')
    parser.add_argument('-z', type=str, default='', help='Copy used audio files directly to this directory, if specified')

    args = parser.parse_args()
    if args.d is None:
        source_dir = os.path.join(source_dir_env, args.c)
    else:
        source_dir = args.d

    start_time = time.time()

    Main(args.a, args.b, args.c, source_dir, args.f, args.i, args.s, args.t, args.z, (args.l == 1)).run()

    elapsed = time.time() - start_time
    print(f'elapsed seconds = {elapsed:.3f}')