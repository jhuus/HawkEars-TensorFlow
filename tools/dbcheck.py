# Check the database integrity.
# This consists of ensuring that all ID references are valid and checking for duplicates.
# *** If a recording is found with no spectrograms, it is deleted. ***

import os
import inspect
import argparse
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import database
from core import util

class Main:
    def __init__(self, dbname, skip_source):
        self.db = database.Database(filename=f'../data/{dbname}.db')
        self.skip_source = skip_source == 1

    def _check_subcategories(self):
        results = self.db.get_all_subcategories()
        self.subcategory_refs = {} # count of refs to each subcategory ID
        self.subcategory_names = {} # map from ID to name
        num_bad_category_refs = 0

        names = {}
        num_duplicate_names = 0

        self.category_map = {} # map subcategory to category
        for result in results:
            id, name, code, synonym, category_id = result
            self.category_map[id] = category_id

            if name in names:
                print(f'Error: duplicate subcategory name {name}')
                num_duplicate_names += 1

            names[name] = 1
            self.subcategory_refs[id] = 0
            self.subcategory_names[id] = name
            if category_id in self.category_refs:
                self.category_refs[category_id] += 1
            else:
                num_bad_category_refs += 1
                print(f'Error: subcategory ID {id} references unknown category ID {category_id}')

        if num_bad_category_refs == 0:
            print('All category ID references in subcategory table are valid')

        if num_duplicate_names == 0:
            print('No duplicate names in subcategory table')

    def _check_recordings(self):
        ignore_list = util.get_file_lines('../data/ignore.txt')
        results = self.db.get_all_recordings()
        self.recording_refs = {} # count of refs to each recording ID
        num_bad_source_refs = 0
        num_bad_subcategory_refs = 0

        key_hash = {}
        num_duplicate_filenames = 0

        for result in results:
            id, source_id, subcategory_id, filename, seconds = result
            subcategory_name = self.subcategory_names[subcategory_id]

            # duplicate filename is allowed, but not for same subcategory
            key = f'{filename}:{subcategory_id}'
            if key in key_hash.keys():
                print(f'Warning: duplicate recording {filename} in subcategory {subcategory_id} ({subcategory_name})')
                num_duplicate_filenames += 1
                self._merge_duplicate_recordings(subcategory_id, filename, id, key_hash[key])
                continue

            key_hash[key] = id
            self.recording_refs[id] = 0
            if source_id in self.source_refs:
                self.source_refs[source_id] += 1
            else:
                num_bad_source_refs += 1
                print(f'Error: recording ID {id} references unknown source ID {source_id}')

            if subcategory_id in self.subcategory_refs:
                self.subcategory_refs[subcategory_id] += 1
            else:
                num_bad_subcategory_refs += 1
                print(f'Error: recording ID {id} references unknown subcategory ID {subcategory_id}')

            if self.category_map[subcategory_id] == self.bird_category_id and not self.skip_source:
                if subcategory_name == 'Rooster':
                    source_name = 'Youtube' # special case
                else:
                    source_name = util.get_source_name(filename)

                if source_id != self.source_ids[source_name]:
                    print(f'Warning: incorrect source for {filename} ({subcategory_name}). It is {self.source_names[source_id]} but should be {source_name}.')
                    self.db.update_recording_source_id(id, self.source_ids[source_name])

        if num_bad_source_refs == 0:
            print('All source ID references in recording table are valid')

        if num_bad_subcategory_refs == 0:
            print('All subcategory ID references in recording table are valid')

        if num_duplicate_filenames == 0:
            print('No duplicate file names in recording table')

    def _check_spectrograms(self):
        results = self.db.get_all_spectrograms()
        num_bad_recording_refs = 0

        spec_names = {}
        num_duplicate_specs = 0

        for result in results:
            id, recording_id, value, offset = result

            spec_name = f'{recording_id}-{offset}'
            if spec_name in spec_names:
                print(f'Error: duplicate spectrogram {spec_name}')
                num_duplicate_specs += 1
                self.db.delete_spectrogram_by_id(id)
            else:
                spec_names[spec_name] = 1

            if recording_id in self.recording_refs:
                self.recording_refs[recording_id] += 1
            else:
                num_bad_recording_refs += 1
                print(f'Error: spectrogram ID {id} references unknown recording ID {recording_id}')

        if num_duplicate_specs == 0:
            print('No duplicate spectrograms')

        if num_bad_recording_refs == 0:
            print('All recording ID references in spectrogram table are valid')

    def _check_no_references(self):
        num_no_refs = 0
        for id in self.source_refs.keys():
            if self.source_refs[id] == 0:
                print(f'Warning: source ID {id} has no references')
                num_no_refs += 1

        if num_no_refs == 0:
            print('All source records have some references')

        num_no_refs = 0
        for id in self.category_refs.keys():
            if self.category_refs[id] == 0:
                print(f'Warning: category ID {id} has no references')
                num_no_refs += 1

        if num_no_refs == 0:
            print('All category records have some references')

        num_no_refs = 0
        for id in self.subcategory_refs.keys():
            if self.subcategory_refs[id] == 0:
                print(f'Warning: subcategory ID {id} has no references')
                num_no_refs += 1

        if num_no_refs == 0:
            print('All subcategory records have some references')

        num_no_refs = 0
        for id in self.recording_refs.keys():
            if self.recording_refs[id] == 0:
                print(f'Warning: recording ID {id} has no references')
                self.db.delete_recording_by_id(id)
                num_no_refs += 1

        if num_no_refs == 0:
            print('All recording records have some references')

    def _merge_duplicate_recordings(self, subcategory_id, filename, id1, id2):
        if filename.startswith('HNC'):
            source_name = 'HNC'
        elif filename.startswith('XC'):
            source_name = 'Xeno-Canto'
        elif filename[0].isdigit():
            source_name = 'Macaulay Library'
        elif filename[0].isalpha():
            source_name = 'Cornell Guide'
        else:
            source_name = 'Youtube'

        correct_source_id = self.source_ids[source_name]
        source_id1, _, _, _, seconds1, _, _, _, _, _, _, _, _, _, _ = self.db.get_recording(id1)
        source_id2, _, _, _, seconds2, _, _, _, _, _, _, _, _, _, _ = self.db.get_recording(id2)

        if source_id1 == correct_source_id:
            source_id = id2
            target_id = id1
        else:
            source_id = id1
            target_id = id2

        source_spectrograms = self.db.get_spectrograms_by_recording_id2(source_id)
        target_spectrograms = self.db.get_spectrograms_by_recording_id2(target_id)

        # compare source_spectrograms with target_spectrograms by offset;
        # if they have a match, delete the source spectrogram, otherwise point it to the target recording
        target_offsets = {}
        for result in target_spectrograms:
            id, offset = result
            target_offsets[offset] = 1

        for result in source_spectrograms:
            id, offset = result
            if offset in target_offsets:
                self.db.delete_spectrogram_by_id(id)
            else:
                self.db.update_spectrogram_recording_id(id, target_id)

        self.db.delete_recording_by_id(source_id)

    def run(self):
        results = self.db.get_all_sources()
        self.source_refs = {} # count of refs to each source ID
        self.source_ids = {} # source ID per name
        self.source_names = {} # source name per ID
        for result in results:
            id, name = result
            self.source_refs[id] = 0
            self.source_ids[name] = id
            self.source_names[id] = name

        results = self.db.get_all_categories()
        self.bird_category_id = 0
        self.category_refs = {} # count of refs to each category ID
        for result in results:
            id, name = result
            self.category_refs[id] = 0
            if name == 'bird':
                self.bird_category_id = id

        self._check_subcategories()
        self._check_recordings()
        self._check_spectrograms()
        self._check_no_references()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='training', help='Database name. Default = training')
    parser.add_argument('-x', type=int, default=0, help='1 = skip source checks. Default = 0.')
    args = parser.parse_args()

    Main(args.f, args.x).run()
