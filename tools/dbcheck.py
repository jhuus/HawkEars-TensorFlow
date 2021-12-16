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

class Main:
    def __init__(self, dbname):
        self.db = database.Database(filename=f'../data/{dbname}.db')
        
    def _check_subcategories(self):
        results = self.db.get_all_subcategories()
        self.subcategory_refs = {} # count of refs to each subcategory ID
        num_bad_category_refs = 0
        
        names = {}
        num_duplicate_names = 0
        
        for result in results:
            id, name, code, synonym, category_id = result
            
            if name in names:
                print(f'Error: duplicate subcategory name {name}')
                num_duplicate_names += 1
            
            names[name] = 1
            self.subcategory_refs[id] = 0
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
        results = self.db.get_all_recordings()
        self.recording_refs = {} # count of refs to each recording ID
        num_bad_source_refs = 0
        num_bad_subcategory_refs = 0
        
        filenames = {}
        num_duplicate_filenames = 0

        for result in results:
            id, source_id, subcategory_id, filename, seconds = result
            
            if filename in filenames:
                print(f'Warning: duplicate recording file {filename}')
                num_duplicate_filenames += 1
            else:
                filenames[filename] = 1
            
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
        
    def run(self):
        results = self.db.get_all_sources()
        self.source_refs = {} # count of refs to each source ID
        for result in results:
            id, name = result
            self.source_refs[id] = 0

        results = self.db.get_all_categories()
        self.category_refs = {} # count of refs to each category ID
        for result in results:
            id, name = result
            self.category_refs[id] = 0

        self._check_subcategories()
        self._check_recordings()
        self._check_spectrograms()
        self._check_no_references()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='training', help='Database name. Default = training')
    args = parser.parse_args()

    Main(args.f).run()
