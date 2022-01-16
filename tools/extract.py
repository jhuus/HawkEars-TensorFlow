# This script has three modes:
#
#     m=0: generate labels
#     m=1: plot spectrograms
#     m=2: import spectrograms for one species (e.g. folder \dbdata\bird-xc\EATO)
#     m=3: import validation spectrograms for multiple species in a single folder
#          (e.g. /dbdata/test contains mp3 files with species code prefixes).
#          This is used when importing separate validation data for use during training, 
#          so that validation data isn't too strongly correlated with training data.
#          In this case file ../data/codes.txt is expected to map each code to its name.
#     m=4: import spectrograms when species names are given in label files.
#
# If specs.txt exists in the given folder, it is used instead of the label files.
# The normal process is:
#
#     1) Run with m=0 to generate labels.
#     2) Run with m=1 to generate spectrograms.
#     3) Create specs.txt by running sortdir.py -d ... -o ...\specs.txt.
#     4) Review spectrograms and edit specs.txt to remove bad ones or shift them a little.
#     5) Delete spectrograms, run m=1 again and review.
#     6) Run m=2, m=3 or m=4 to add spectrograms to database.
#

import argparse
import inspect
import os
import random
import sys
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import audio
from core import constants
from core import database
from core import util

class Spectrogram:
    def __init__(self, spec, name, recording_id, offset):
        self.spec = spec
        self.name = name
        self.recording_id = recording_id
        self.offset = offset

class Main:
    def __init__(self, mode, label_type, root, source, category, subcategory, code, dbname, sound_factor, binary_classifier, keep_empty, sparse_noise):
        self.mode = mode
        self.db = database.Database(filename=f'../data/{dbname}.db')
        self.audio = audio.Audio(path_prefix='../')
        
        self.label_type = label_type
        self.root = root
        self.source = source
        self.category = category
        self.subcategory = subcategory
        self.code = code
        self.sound_factor = sound_factor
        self.binary_classifier = (binary_classifier == 1)
        self.keep_empty = (keep_empty == 1)
        self.sparse_noise = (sparse_noise == 1)
        
    def _fatal_error(self, message):
        print(message)
        sys.exit()
        
    # generate sparse noise spectrograms by stripping most of the sound from the input spectrograms
    # and randomly combining spectrograms
    def _convert_to_sparse_noise(self, specs):
        min_val, max_val = 0.8, 0.9
        noise_specs = []
        for i in range(len(specs)):
            spec = np.clip(specs[i], min_val, max_val) / max_val
            used_so_far = [i]
            num_specs = random.randint(1, 3)
            for j in range(num_specs):
                spec_num = random.randint(0, len(specs) - 1)
                while spec_num in used_so_far:
                    spec_num = random.randint(0, len(specs) - 1)

                spec += np.clip(specs[spec_num], min_val, max_val) / max_val
                used_so_far.append(spec_num)

            noise_specs.append(spec / np.max(spec))
        
        return noise_specs 
        
    def _generate_labels(self):
        print(f'generate labels')
        if not os.path.exists(self.label_path):
            os.makedirs(self.label_path)

        audio_files = util.get_audio_files(self.root)
        for file_path in audio_files:
            print(f'processing {file_path}')
            signal, rate = self.audio.load(file_path)

            if self.label_type == 0:
                starts = self.audio.find_sounds(sound_factor=self.sound_factor, keep_empty=self.keep_empty)
            else:
                seconds = int(len(signal) / rate)
                starts = [i for i in range(0, seconds, 3)]
            
            basename = os.path.basename(file_path)
            prefix = basename.split('.')[-2]
            path = f'{self.label_path}/{prefix}.txt'
            with open(path, 'w') as file:
                for start in starts:
                    if rate * (start + 3) <= len(signal):
                        file.write(f'{start:.4f}\t{(start+3):.4f}\tsound\n')
                    
            file.close()

    # for mode 3 we need to map species codes to names, using data/codes.txt
    def _get_code_dict(self):
        lines = util.get_file_lines(r'../data/codes.txt')
        code_dict = {} # map species codes to names, e.g. "AMRO" => "American Robin"
        for line in lines:
            tokens = line.split(',')
            code_dict[tokens[0]] = tokens[1]
            
        return code_dict
        
    def _get_offsets(self, filename):
        base_name = Path(filename).stem
        if base_name in self.offsets_dict.keys():
            return self.offsets_dict[base_name]
        else:
            return []
        
    # get a 3-second spectrogram at each of the given offsets of the specified file
    def _get_spectrograms(self, path, offsets):
        specs = []
        signal, rate = self.audio.load(path)
        seconds = len(signal) / rate

        filename = os.path.basename(path)
        index = filename.rfind('.')
        prefix = filename[:index]
        url = ''

        if self.mode >= 2:
            recording_id = self.db.get_recording_id(self.source_id, self.subcategory_id, filename)
            if recording_id is None:
                recording_id = self.db.insert_recording(self.source_id, self.subcategory_id, url, filename, seconds)
        else:
            recording_id = None

        raw_specs = self.audio.get_spectrograms(offsets, binary_classifier=self.binary_classifier)
        for i in range(len(offsets)):
            specs.append(Spectrogram(raw_specs[i], f'{prefix}-{offsets[i]}', recording_id, offsets[i]))
            
        return specs, raw_specs
        
    # insert spectrograms into the database
    def _import_spectrograms(self):
        print(f'import spectrograms')
        audio_files = util.get_audio_files(self.root)
        specs = []
        raw_specs = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            print(f'processing {filename}')
            offsets = self._get_offsets(filename)
            curr_specs, curr_raw_specs = self._get_spectrograms(audio_file, offsets)
            specs.extend(curr_specs)
            raw_specs.extend(curr_raw_specs)

        if self.sparse_noise:
            raw_specs = self._convert_to_sparse_noise(raw_specs)
            for i in range(len(specs)):
                specs[i].spec = raw_specs[i]
            
        for spec in specs:
            if not spec.spec is None:
                # convert to bytes, zip it and insert in database
                compressed = util.compress_spectrogram(spec.spec)
                self.db.insert_spectrogram(spec.recording_id, compressed, spec.offset)
        
    def _plot_spectrograms(self):
        print(f'plot spectrograms')
        if not os.path.exists(self.spec_path):
            os.makedirs(self.spec_path)
    
        audio_files = util.get_audio_files(self.root)
        specs = []
        raw_specs = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            print(f'processing {filename}')
            offsets = self._get_offsets(filename)
            curr_specs, curr_raw_specs = self._get_spectrograms(audio_file, offsets)
            specs.extend(curr_specs)
            raw_specs.extend(curr_raw_specs)

        if self.sparse_noise:
            raw_specs = self._convert_to_sparse_noise(raw_specs)
            for i in range(len(specs)):
                specs[i].spec = raw_specs[i]
        
        for spec in specs:
            util.plot_spec(spec.spec, os.path.join(self.spec_path, f'{spec.name}.png'), binary_classifier=self.binary_classifier)
        
    # In this case (m=3) the root folder contains audio files for multiple species, where each file name
    # has the species code as a prefix (e.g. BAWW_XC213571.mp3).
    def _import_validation(self):
        print(f'import validation spectrograms')
        code_dict = self._get_code_dict()
        audio_files = util.get_audio_files(self.root)
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            print(f'processing {filename}')
            
            code = filename[:4]
            if code in code_dict.keys():
                species_name = code_dict[code]
            else:
                self._fatal_error(f'Error: file {filename} does not start with a valid species code.')
                
            result = self.db.get_subcategory(self.category_id, species_name)
            if result is None:
                self.subcategory_id = self.db.insert_subcategory(self.category_id, species_name, code=code)
            else:
                (self.subcategory_id, _) = result
            
            offsets = self._get_offsets(filename)
            specs = self._get_spectrograms(audio_file, offsets)
            
            for spec in specs:
                if not spec.spec is None:
                    # convert to bytes, zip it and insert in database
                    compressed = util.compress_spectrogram(spec.spec)
                    self.db.insert_spectrogram(spec.recording_id, compressed, spec.offset)
                
    # when label files identify the species (mode 4), import all species for one label file
    def _import_labelled_species(self, file_path, label_filepath, audio_files):
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            label_filename = os.path.basename(label_filepath)
            if filename.startswith(label_filename):
                print(f'processing file {filename}')
                for species_name in self.species_dict.keys():
                    print(f'processing subcategory {species_name}')

                    result = self.db.get_subcategory_by_name(species_name)
                    if result is None:
                        self._fatal_error(f'No subcategory record found for {species_name}. In mode 4, subcategory records must already exist.')
                    else:
                        (self.subcategory_id, _) = result
                        
                    offsets = self.species_dict[species_name]
                    specs = self._get_spectrograms(audio_file, offsets)
            
                    for spec in specs:
                        if not spec.spec is None:
                            result = self.db.get_spectrogram(spec.recording_id, spec.offset)
                            if result is None:
                                compressed = util.compress_spectrogram(spec.spec)
                                self.db.insert_spectrogram(spec.recording_id, compressed, spec.offset)

    def _init_database(self):
        self.source_id = self.db.get_source_by_name(self.source)
        if self.source_id is None:
            self.source_id = self.db.insert_source(self.source)

        self.category_id = self.db.get_category_by_name(self.category)
        if self.category_id is None:
            self.category_id = self.db.insert_category(self.category)
            
        if self.mode == 2:
            result = self.db.get_subcategory(self.category_id, self.subcategory)
            if result is None:
                self.subcategory_id = self.db.insert_subcategory(self.category_id, self.subcategory, code=self.code)
            else:
                (self.subcategory_id, _) = result
                
    # read a label file and update self.offsets_dict
    def _process_label_file(self, label_file_path):
        filename = Path(label_file_path).stem
        for line in util.get_file_lines(label_file_path):
            try:
                tokens = line.strip().split('\t')
                if len(tokens) > 0:
                    offset = tokens[0].strip()
                    if len(offset) > 0 and offset != '\\':
                        suffix = tokens[2].strip()
                        if suffix == 'sound' or suffix.startswith(self.subcategory):
                            if filename not in self.offsets_dict.keys():
                                self.offsets_dict[filename] = []
                                
                            self.offsets_dict[filename].append(float(offset))
            except:
                continue
                
                
    # read a label file and separate the labels per species, for mode=4
    def _process_label_file_with_species(self, label_file_path):
        filename = Path(label_file_path).stem
        for line in util.get_file_lines(label_file_path):
            tokens = line.strip().split('\t')
            if len(tokens) > 0:
                offset = tokens[0].strip()
                if len(offset) > 0 and offset != '\\':
                    suffix = tokens[2].strip()
                    if suffix not in self.species_dict.keys():
                        self.species_dict[suffix] = []

                    self.species_dict[suffix].append(float(offset))
                
    # read specs.txt and update self.offsets_dict
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
                
    def run(self):
        self._init_database()
        self.label_path = os.path.join(self.root, 'labels')
        self.spec_path = os.path.join(self.root, 'spectrograms')

        if self.mode == 4:
            audio_files = util.get_audio_files(self.root)
            if os.path.exists(self.label_path):
                for file_name in os.listdir(self.label_path):
                    file_path = os.path.join(self.label_path, file_name) 
                    if os.path.isfile(file_path):
                        base_name, ext = os.path.splitext(file_path)
                        if ext == '.txt':
                            self.species_dict = {}
                            self._process_label_file_with_species(file_path)
                            self._import_labelled_species(file_path, base_name, audio_files)
        elif self.mode > 0:
            # get label/specs info unless we're generating labels
            self.offsets_dict = {}
            specs_file_path = os.path.join(self.root, 'specs.txt')
            if os.path.isfile(specs_file_path):
                self._process_specs_file(specs_file_path)
            else:
                # no specs.txt, so look for label files
                if os.path.exists(self.label_path):
                    for file_name in os.listdir(self.label_path):
                        file_path = os.path.join(self.label_path, file_name) 
                        if os.path.isfile(file_path):
                            base, ext = os.path.splitext(file_path)
                            if ext == '.txt':
                                self._process_label_file(file_path)
        
        if self.mode == 0:
            self._generate_labels()
        elif self.mode == 1:
            self._plot_spectrograms()
        elif self.mode == 2:
            self._import_spectrograms()
        elif self.mode == 3:
            self._import_validation()
            
        self.db.close()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=2, help='Mode. 0 = generate labels, 1 = plot spectrograms, 2 = import spectrograms (see comments at top of script). Default = 2.')
    parser.add_argument('-t', type=int, default=0, help='Label type for mode = 0. type 0 means generate labels at sounds, type 1 means generate a label every 3 seconds.')
    parser.add_argument('-d', type=str, default='', help='Directory containing the audio files.')
    parser.add_argument('-a', type=str, default='Xeno-Canto', help='Source (e.g. "Xeno-Canto").')
    parser.add_argument('-b', type=str, default='bird', help='Category (e.g. bird).')
    parser.add_argument('-s', type=str, default='', help='Subcategory (e.g. "Baltimore Oriole").')
    parser.add_argument('-c', type=str, default='', help='Code (e.g. BAOR).')
    parser.add_argument('-f', type=str, default='training', help='Database name. Default = training')
    parser.add_argument('-g', type=float, default=2.5, help='Factor for call to find_sounds when t = 0. Default = 2.5')
    parser.add_argument('-e', type=int, default=0, help='If e = 1, generate labels even if empty (no apparent sound). Default = 0.')
    parser.add_argument('-n', type=int, default=0, help='If n = 1, convert spectrograms to sparse noise. Default = 0.')
    parser.add_argument('-x', type=int, default=0, help='If x = 1, extract spectrograms for binary classifier. Default = 0.')
    
    args = parser.parse_args()

    Main(args.m, args.t, args.d, args.a, args.b, args.s, args.c, args.f, args.g, args.x, args.e, args.n).run()