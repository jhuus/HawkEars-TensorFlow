# Reextract spectrograms for the given species.
# Check source_db to find file names and offsets, then find the files and
# import the matching offsets to target_db.
# This is sometimes necessary when the spectrogram creation logic changes.

import argparse
import inspect
import logging
import os
import shutil
import sys
import time
import zlib

import numpy as np

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import audio
from core import database
from core import util

# extract all the spectrograms for the given recording and attach to the recording object
def extract_spectrograms(recording, audio_object, subcategory, low_noise_detector, from3to4):
    recording.specs = []
    if len(recording.offsets) > 0:
        signal, rate = audio_object.load(recording.file_path)
        if signal is None:
            return

        seconds = len(signal) / rate

        offsets = []
        for offset in recording.offsets:
            if from3to4 and offset >= 0.5:
                offset -= 0.5 # back up half a second when converting from 3 to 4 seconds

            offsets.append(offset)

        if subcategory == 'Noise':
            check_noise = False
            reduce_noise = False
        else:
            check_noise = True
            reduce_noise = True

        specs = audio_object.get_spectrograms(offsets, check_noise=check_noise, reduce_noise=reduce_noise, multi_spec=True, low_noise_detector=low_noise_detector)

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
    def __init__(self, mode, root1, root2, subcategory, prefix, offset, source_db, target_db, low_noise_detector, from3to4):
        self.mode = mode
        self.root1 = root1
        self.root2 = root2
        self.subcategory = subcategory
        self.prefix = prefix.lower()
        self.offset = offset
        self.source_db = database.Database(f'../data/{source_db}.db')
        self.target_db = database.Database(f'../data/{target_db}.db')
        self.low_noise_detector = (low_noise_detector == 1)
        self.from3to4 = (from3to4 == 1)

        if self.low_noise_detector:
            self.root1 = os.path.join(self.root1, 'binary')

    def fatal_error(self, message):
        logging.error(message)
        sys.exit()

    # convert the given spectrogram to bytes, zip it and insert in database
    def insert_spectrogram(self, spec):
        compressed = util.compress_spectrogram(spec.data)
        self.target_db.insert_spectrogram(spec.recording.target_id, compressed, spec.offset)
        self.inserted_spectrograms += 1

    def get_recordings(self):
        cat_path = os.path.join(self.root1, self.cat_name)
        hnc_path = os.path.join(self.root1, 'hnc')
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
            elif not file_path.endswith('gennoise.py'):
                source_path = os.path.join(os.path.join(self.root2, self.code), file_name)
                if os.path.exists(source_path):
                    logging.info(f'Copying {source_path} to {file_path}')
                    shutil.copyfile(source_path, file_path)
                else:
                    logging.error(f'Recording not found: {file_path}')

        return

    def process_recordings(self, source_subcat_id, target_subcat_id):
        source_dict = {} # map source name to target source ID

        # insert recording records
        for recording in self.recordings:
            result = self.source_db.get_recording(recording.source_id)
            source_source_id, source_subcategory_id, url, filename, seconds, recorder, license, quality, latitude, longitude, sound_type, time, date, remark, was_it_seen = result

            source_name = self.source_db.get_source_by_id(source_source_id)
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

        # extract spectrograms
        logging.info(f'Extracting spectrograms for {self.subcategory}')
        spec = []

        audio_object = audio.Audio(path_prefix='../')
        for recording in self.recordings:
            logging.info(f'Processing {recording.file_name}')
            if self.offset is None:
                offsets = self.source_db.get_spectrogram_offsets_by_recording_id(recording.source_id)
                recording.offsets = []
                for result in offsets:
                    recording.offsets.append(result[0])
            else:
                recording.offsets = [self.offset]

            extract_spectrograms(recording, audio_object, self.subcategory, self.low_noise_detector, self.from3to4)

        # insert spectrograms into the database
        logging.info(f'Inserting spectrograms for {self.subcategory}')
        for recording in self.recordings:
            for spec in recording.specs:
                self.insert_spectrogram(spec)

    def run(self):
        start_time = time.time()

        if self.subcategory is None:
            results = self.source_db.get_all_subcategories()
            for result in results:
                _, self.subcategory, _, _, _ = result
                self.do_one()
        else:
            self.do_one()

        elapsed = time.time() - start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        logging.info(f'Elapsed time = {minutes}m {seconds}s\n')

    def do_one(self):
        logging.info(f'Initializing database for {self.subcategory}')
        result = self.source_db.get_subcategory_details_by_name(self.subcategory)
        if result is None:
            self.fatal_error(f'Subcategory {self.subcategory} not found in source database')

        source_subcat_id, source_cat_id, code, synonym, _, _ = result
        self.code = code.strip()

        self.cat_name = self.source_db.get_category_by_id(source_cat_id)
        self.get_recordings()
        if self.mode == 0:
            return

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
        logging.info(f'Inserted {self.inserted_spectrograms} spectrograms')

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-lnd', type=int, default=0, help='1 = spectrograms for low noise detector. Default = 0.')
    parser.add_argument('-d1', type=str, default='training', help='Source database name. Default = training')
    parser.add_argument('-d2', type=str, default='training2', help='Target database name. Default = training2')
    parser.add_argument('-m', type=int, default=0, help='Mode where 0 means just check file availability and 1 means also extract spectrograms. Default = 0.')
    parser.add_argument('-r1', type=str, default='/home/jhuus/data', help='Root directory containing bird/BLJA etc. Default is /home/jhuus/data')
    parser.add_argument('-r2', type=str, default='/media/jhuus/Data/bird', help='Root directory to copy missing files from.')
    parser.add_argument('-s', type=str, default=None, help='Species or subcategory name. If omitted, do all subcategories.')
    parser.add_argument('-p', type=str, default='', help='Only extract from file names having this prefix (case-insensitive). Default is empty, which means extract all.')
    parser.add_argument('-o', type=float, default=None, help='Only extract at this offset (for testing / debugging).')
    parser.add_argument('-z', type=int, default=0, help='z=1 means resample from 3 seconds to 4 seconds. Default = 0.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

    Main(args.m, args.r1, args.r2, args.s, args.p, args.o, args.d1, args.d2, args.lnd, args.z).run()
