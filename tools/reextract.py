# Reextract spectrograms for the given species. 
# Check source_db to find file names and offsets, then find the files and 
# import the matching offsets to target_db. 
# This is sometimes necessary when the spectrogram creation logic changes. 

import argparse
import inspect
import os
import sys
import threading
import time
import zlib

import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import audio
from core import constants
from core import database
from core import util

# thread function to extract all the spectrograms for the given recording
def extract_spectrograms(recording, audio_object, binary_classifier):
    recording.specs = []
    if len(recording.offsets) > 0:
        signal, rate = audio_object.load(recording.file_path)
        seconds = len(signal) / rate
        
        offsets = []
        for result in recording.offsets:
            offsets.append(result[0])
            
        specs = audio_object.get_spectrograms(offsets, binary_classifier=binary_classifier)
        for i in range(len(offsets)):
            if not specs[i] is None:
                recording.specs.append(Spectrogram(recording, offsets[i], specs[i]))

class Recording:
    def __init__(self, source_id, file_name, file_path):
        self.source_id = source_id
        self.file_name = file_name
        self.file_path = file_path
        
class Spectrogram:
    def __init__(self, recording, offset, data):
        self.recording = recording
        self.offset = offset
        self.data = data

class Main:
    def __init__(self, mode, root, subcategory, prefix, source_db, target_db, num_threads, binary_classifier):
        self.mode = mode
        self.root = root
        self.subcategory = subcategory
        self.prefix = prefix.lower()
        self.source_db = database.Database(f'../data/{source_db}.db')
        self.target_db = database.Database(f'../data/{target_db}.db')
        self.num_threads = num_threads
        self.binary_classifier = (binary_classifier == 1)

        if self.binary_classifier:
            self.root = os.path.join(self.root, 'binary')
        
    def fatal_error(self, message):
        print(message)
        sys.exit()
                    
    # convert the given spectrogram to bytes, zip it and insert in database
    def insert_spectrogram(self, spec):
        compressed = util.compress_spectrogram(spec.data)
        self.target_db.insert_spectrogram(spec.recording.target_id, compressed, spec.offset)
        self.inserted_spectrograms += 1
        
    def get_recordings(self):
        cat_path = os.path.join(self.root, self.cat_name)
        hnc_path = os.path.join(self.root, 'hnc')
        if len(self.code) == 0:
            subcat_path = os.path.join(cat_path, self.subcategory)
        else:
            subcat_path = os.path.join(cat_path, self.code)
    
        self.recordings = []
        results = self.source_db.get_recordings_by_subcategory_name(self.subcategory)
        for result in results:
            recording_id, file_name, _ = result
            file_name_lower = file_name.lower()
            if len(self.prefix) > 0 and not file_name_lower.startswith(self.prefix):
                continue
            
            if file_name_lower.startswith('hnc'):
                file_path = os.path.join(hnc_path, file_name)
            else:
                file_path = os.path.join(subcat_path, file_name)
            
            found = True
            if not os.path.exists(file_path):
                found = False
                suffixes = ['.wav', '.octet-stream']
                for suffix in suffixes:
                    if not found and file_path.lower().endswith(suffix):
                        # try mp3 in case it was converted
                        file_path = file_path[:-len(suffix)] + '.mp3'
                        if os.path.exists(file_path):
                            file_name = file_name[:-4] + '.mp3'
                            found = True

            if found:
                self.recordings.append(Recording(recording_id, file_name, file_path))
            else:
                print(f'Recording not found: {file_path}')
            
        return
        
    def process_recordings(self, source_subcat_id, target_subcat_id):
        source_dict = {} # map source name to target source ID
        
        # insert recording records, which must be single-threaded
        for recording in self.recordings:
            result = self.source_db.get_recording(recording.source_id)
            source_source_id, source_subcategory_id, url, filename, seconds, recorder, license, quality, latitude, longitude, sound_type, time, date, remark, was_it_seen = result
        
            source_name = self.source_db.get_source_by_id(source_source_id)
            if recording.file_name.lower().startswith('hnc'):
                source_name = 'HNC' # fix a consistency issue
            
            if source_name in source_dict.keys():
                target_source_id = source_dict[source_name]
            else:
                target_source_id = self.target_db.get_source_by_name(source_name)
                if target_source_id is None:
                    target_source_id = self.target_db.insert_source(source_name)
                
                source_dict[source_name] = target_source_id
             
            recording.target_id = self.target_db.get_recording_id(target_source_id, target_subcat_id, recording.file_name)
            if recording.target_id is None:
                recording.target_id = self.target_db.insert_recording(target_source_id, target_subcat_id, url, recording.file_name, seconds=seconds, recorder=recorder, license=license,
                             quality=quality, latitude=latitude, longitude=longitude, sound_type=sound_type, time=time, date=date, remark=remark, was_it_seen=was_it_seen)
            
        # extract spectrograms, which is multi-threaded
        print(f'Extracting spectrograms for {self.subcategory}')
        spec = []
        rec_num = 0
        thread_num = 0
        
        # create an audio object per thread
        audio_objects = []
        for thread_num in range(self.num_threads):
            audio_objects.append(audio.Audio(path_prefix='../'))
        
        while rec_num < len(self.recordings):
            # start threads
            thread_specs = {}
            threads = []
            for thread_num in range(self.num_threads):
                if rec_num < len(self.recordings):
                    recording = self.recordings[rec_num]
                    print(f'Processing {recording.file_name}')
                    recording.offsets = self.source_db.get_spectrogram_offsets_by_recording_id(recording.source_id)
                    
                    thread = threading.Thread(target=extract_spectrograms, args=(recording, audio_objects[thread_num], self.binary_classifier))
                    thread.start()
                    threads.append(thread)
                    rec_num += 1
            
            # wait for threads to finish
            for thread in threads:
                thread.join()
            
        # insert spectrograms into the database, which must be single-threaded
        print(f'Inserting spectrograms for {self.subcategory}')
        for recording in self.recordings:
            for spec in recording.specs:
                self.insert_spectrogram(spec)
        
    def run(self):
        start_time = time.time()
        print(f'Initializing database for {self.subcategory}')
        result = self.source_db.get_subcategory_details_by_name(self.subcategory)
        if result is None:
            self.fatal_error(f'Subcategory {self.subcategory} not found in source database')

        source_subcat_id, source_cat_id, code, synonym, _, _ = result
        self.code = code.strip()
        
        self.cat_name = self.source_db.get_category_by_id(source_cat_id)
        self.get_recordings()
        if self.mode == 0:
            sys.exit()
        
        target_cat_id = self.target_db.get_category_by_name(self.cat_name)
        if target_cat_id is None:
            target_cat_id = self.target_db.insert_category(self.cat_name)
        
        result = self.target_db.get_subcategory_by_name(self.subcategory)
        if result is None:
            target_subcat_id = self.target_db.insert_subcategory(target_cat_id, self.subcategory, synonym=synonym, code=self.code)
        else:
            target_subcat_id, _ = result
            
        self.inserted_spectrograms = 0
        self.process_recordings(source_subcat_id, target_subcat_id)
        
        elapsed = time.time() - start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        print(f'Elapsed time to insert {self.inserted_spectrograms} spectrograms = {minutes}m {seconds}s\n')

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=0, help='1 = spectrograms for binary classifier. Default = 0.')
    parser.add_argument('-d1', type=str, default='training', help='Source database name. Default = training')
    parser.add_argument('-d2', type=str, default='training2', help='Target database name. Default = training2')
    parser.add_argument('-m', type=int, default=0, help='Mode where 0 means just check file availability and 1 means also extract spectrograms. Default = 0.')
    parser.add_argument('-r', type=str, default='/home/jhuus/data', help='Root directory containing bird/BLJA etc. Default is /home/jhuus/data')
    parser.add_argument('-s', type=str, default='', help='Species or subcategory name.')
    parser.add_argument('-p', type=str, default='', help='Only extract from file names having this prefix (case-insensitive). Default is empty, which means extract all.')
    parser.add_argument('-t', type=int, default=1, help='Number of threads. Default = 1.')
    
    args = parser.parse_args()

    Main(args.m, args.r, args.s, args.p, args.d1, args.d2, args.t, args.b).run()
    