# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and output an Audacity label file
# with the class predictions.

import argparse
import logging
import os
import re
import sys
import time
import warnings

warnings.filterwarnings('ignore') # suppress librosa warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import tensorflow as tf
from tensorflow import keras

from core import audio
from core import config as cfg
#from core import frequency_db
from core import plot
from core import util

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.frequency_too_low = False
        self.reset()

    def reset(self):
        self.has_label = False
        self.probs = [] # predictions (one per segment)
        self.frequency_too_low = False

class Label:
    def __init__(self, class_name, probability, start_time, end_time):
        self.class_name = class_name
        self.probability = probability
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, input_path, output_path, start_time, end_time, date_str, latitude, longitude, debug_mode):

        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.date_str = date_str
        self.latitude = latitude
        self.longitude = longitude
        self.debug_mode = (debug_mode == 1)

        if self.start_seconds is not None and self.end_seconds is not None and self.end_seconds < self.start_seconds + cfg.segment_len:
                logging.error(f'Error: end time must be >= start time + {cfg.segment_len} seconds')
                quit()

        if self.end_seconds is not None:
            # convert from end of last segment to start of last segment for processing
            self.end_seconds = max(0, self.end_seconds - cfg.segment_len)

        # if no output path specified and input path is a directory,
        # put the output labels in the input directory
        if len(self.output_path) == 0:
            if os.path.isdir(self.input_path):
                self.output_path = self.input_path
        elif not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.class_infos = self._get_class_infos()
        self.audio = audio.Audio()
        self._process_lat_lon_date()

    # process latitude, longitude and date
    def _process_lat_lon_date(self):
        if self.latitude is None and self.longitude is None and self.date_str is None:
            self.check_frequency = False
            return

        '''
        self.check_frequency = True
        if self.latitude is None or self.longitude is None or self.date_str is None:
            logging.error(f'Error: if latitude, longitude or date are specified, all three must be specified')
            quit()

        if self.date_str == 'file':
            self.get_date_from_file_name = True
        else:
            self.get_date_from_file_name = False
            self.week_num = self._get_week_num_from_date_str(self.date_str)
            if self.week_num is None:
                logging.error(f'Error: invalid date string: {self.date_str}')
                quit()

        freq_db = frequency_db.Frequency_DB()
        self.counties = freq_db.get_all_counties()

        county = None
        for c in self.counties:
            if self.latitude >= c.min_y and self.latitude <= c.max_y and self.longitude >= c.min_x and self.longitude <= c.max_x:
                county = c
                break

        if county is None:
            logging.error(f'Error: no eBird county found matching given latitude and longitude')
            quit()
        else:
            logging.info(f'Matching species in {county.name} ({county.code})')

        for class_info in self.class_infos:
            if not class_info.ignore:
                results = freq_db.get_frequencies(county.id, class_info.name)
                class_info.frequency_dict = {}
                for result in results:
                    class_info.frequency_dict[result.week_num] = result.value
        '''

    # return week number in the range [1, 48] as used by eBird barcharts, i.e. 4 weeks per month
    def _get_week_num_from_date_str(self, date_str):
        if not date_str.isnumeric():
            return None

        if len(date_str) >= 4:
            month = int(date_str[-4:-2])
            day = int(date_str[-2:])
            week_num = (month - 1) * 4 + min(4, (day - 1) // 7 + 1)
            return week_num
        else:
            return None

    def _get_seconds_from_time_string(self, time_str):
        time_str = time_str.strip()
        if len(time_str) == 0:
            return None

        seconds = 0
        tokens = time_str.split(':')
        if len(tokens) > 2:
            seconds += 3600 * int(tokens[-3])

        if len(tokens) > 1:
            seconds += 60 * int(tokens[-2])

        seconds += float(tokens[-1])
        return seconds

    def _get_class_infos(self):
        classes = util.get_class_list(class_file_path=cfg.classes_file)
        class_dict = util.get_class_dict(class_file_path=cfg.classes_file)
        ignore_list = util.get_file_lines(cfg.ignore_file)

        class_infos = []
        for class_name in classes:
            if class_name in ignore_list:
                ignore = True
            else:
                ignore = False

            class_infos.append(ClassInfo(class_name, class_dict[class_name], ignore))

        return class_infos

    def _get_file_list(self):
        if os.path.isdir(self.input_path):
            return util.get_audio_files(self.input_path)
        elif util.is_audio_file(self.input_path):
            return [self.input_path]
        else:
            logging.error(f'Error: {self.input_path} is not a directory or an audio file.')
            sys.exit()

    def _get_predictions(self, signal, rate):
        # if needed, pad the signal with zeros to get the last spectrogram

        total_seconds = self.audio.signal_len() / rate
        last_segment_len = total_seconds - cfg.segment_len * (total_seconds // cfg.segment_len)
        if last_segment_len > 0.5:
            # more than 1/2 a second at the end, so we'd better analyze it
            pad_amount = int(rate * (cfg.segment_len - last_segment_len)) + 1
            signal = np.pad(signal, (0, pad_amount), 'constant', constant_values=(0, 0))

        start_seconds = 0 if self.start_seconds is None else self.start_seconds
        if self.debug_mode:
            end_seconds = start_seconds # just do one segment in debug mode
        elif self.end_seconds is None:
            end_seconds = (self.audio.signal_len() / rate) - cfg.segment_len
        else:
            end_seconds = self.end_seconds

        specs = self._get_specs(start_seconds, end_seconds, filter=False, dampen_low_noise=True)
        if cfg.bandpass_filter:
            filtered_specs = self._get_specs(start_seconds, end_seconds, filter=True, dampen_low_noise=False)
        else:
            filtered_specs = None

        logging.info('Done creating spectrograms')

        predictions = self.model.predict(specs)
        if cfg.bandpass_filter:
            filtered_predictions = self.model.predict(filtered_specs)

        if self.debug_mode:
            self._log_predictions(predictions, filtered_predictions)

        # populate class_infos with predictions, using the max with and without bandpass filter
        for i in range(len(self.offsets)):
            for j in range(len(self.class_infos)):
                    value = max(predictions[i][j], filtered_predictions[i][j]) if cfg.bandpass_filter else predictions[i][j]
                    self.class_infos[j].probs.append(value)
                    if (self.class_infos[j].probs[-1] >= cfg.min_prob):
                        self.class_infos[j].has_label = True

    # get the list of spectrograms
    def _get_specs(self, start_seconds, end_seconds, filter, dampen_low_noise):
        self.offsets = np.arange(start_seconds, end_seconds + 1.0, 1.0).tolist()
        specs = self.audio.get_spectrograms(self.offsets, filter=filter, dampen_low_noise=dampen_low_noise)

        spec_array = np.zeros((len(specs), cfg.spec_height, cfg.spec_width, 1))
        for i in range(len(specs)):
            spec_array[i] = specs[i].reshape((cfg.spec_height, cfg.spec_width, 1)).astype(np.float32)

        return spec_array

    def _analyze_file(self, file_path):
        logging.info(f'Analyzing {file_path}')

        check_frequency = False
        if self.check_frequency:
            check_frequency = True
            if self.get_date_from_file_name:
                result = re.split('\S+_(\d+)_.*', os.path.basename(file_path))
                if len(result) > cfg.file_date_regex_group:
                    date_str = result[cfg.file_date_regex_group]
                    self.week_num = self._get_week_num_from_date_str(date_str)
                    if self.week_num is None:
                        logging.error(f'Error: invalid date string: {self.date_str} extracted from {file_path}')
                        check_frequency = False # ignore species frequencies for this file

        # clear info from previous recording, and mark classes where frequency of eBird reports is too low
        for class_info in self.class_infos:
            class_info.reset()
            if check_frequency and not class_info.ignore:
                if not self.week_num in class_info.frequency_dict or class_info.frequency_dict[self.week_num] < cfg.min_freq:
                    class_info.frequency_too_low = True

        signal, rate = self.audio.load(file_path)

        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # generate labels for one class at a time
        labels = []
        min_adj_prob = cfg.min_prob * cfg.adjacent_prob_factor # in mode 0, adjacent segments need this prob at least

        for class_info in self.class_infos:
            if class_info.ignore or class_info.frequency_too_low or not class_info.has_label:
                continue

            if cfg.use_banding_codes:
                name = class_info.code
            else:
                name = class_info.name

            prev_label = None
            probs = class_info.probs
            for i in range(len(probs)):

                # create a label if probability exceeds the threshold
                if probs[i] >= cfg.min_prob:
                    use_prob = probs[i]
                else:
                    continue

                end_time = self.offsets[i]+cfg.segment_len
                if cfg.check_adjacent:
                    if i not in [0, len(probs) - 1]:
                        if probs[i - 1] < min_adj_prob and probs[i + 1] < min_adj_prob:
                            continue

                    if prev_label != None and prev_label.end_time >= self.offsets[i]:
                        # extend the previous label's end time (i.e. merge)
                        prev_label.end_time = end_time
                        prev_label.probability = max(use_prob, prev_label.probability)
                    else:
                        label = Label(name, use_prob, self.offsets[i], end_time)
                        labels.append(label)
                        prev_label = label
                else:
                    label = Label(name, use_prob, self.offsets[i], end_time)
                    labels.append(label)
                    prev_label = label

        self._save_labels(labels, file_path)

    def _save_labels(self, labels, file_path):
        basename = os.path.basename(file_path)
        tokens = basename.split('.')
        output_path = os.path.join(self.output_path, f'{tokens[0]}_HawkEars_Audacity_Labels.txt')
        logging.info(f'Writing output to {output_path}')
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_name};{label.probability:.2f}\n')

        except:
            logging.error(f'Unable to write file {output_path}')
            sys.exit()

    # in debug mode, output the top predictions
    def _log_predictions(self, predictions, filtered_predictions):
        predictions = np.copy(predictions[0])
        print("\ntop predictions")
        for i in range(cfg.top_n):
            j = np.argmax(predictions)
            code = self.class_infos[j].code
            confidence = predictions[j]
            print(f"{code}: {confidence}")
            predictions[j] = 0

        if cfg.bandpass_filter:
            print("\ntop predictions on filtered audio")
            for i in range(cfg.top_n):
                j = np.argmax(filtered_predictions)
                code = self.class_infos[j].code
                confidence = filtered_predictions[j]
                print(f"{code}: {confidence}")
                filtered_predictions[j] = 0

        print("")

    def run(self, start_time):
        file_list = self._get_file_list()

        # load the models
        self.model = keras.models.load_model(cfg.ckpt_path, compile=False)

        for i, file_path in enumerate(file_list):
            self._analyze_file(file_path)

            # this helps to avoid running out of memory on GPU
            if (i + 1) % cfg.reset_model_counter == 0:
                keras.backend.clear_session()
                del self.model
                del self.audio
                self.model = keras.models.load_model(cfg.ckpt_path, compile=False)
                self.audio = audio.Audio()

        elapsed = time.time() - start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        logging.info(f'Elapsed time = {minutes}m {seconds}s')

if __name__ == '__main__':
    check_adjacent = 1 if cfg.check_adjacent else 0
    use_banding_codes = 1 if cfg.use_banding_codes else 0

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--adj', type=int, default=check_adjacent, help=f'1 = Ignore if neither adjacent/overlapping spectrogram matches. Default = {check_adjacent}.')
    parser.add_argument('-b', '--band', type=int, default=use_banding_codes, help=f'1 = Use banding codes in labels, 0 = Use species names. Default = {use_banding_codes}.')
    parser.add_argument('-d', '--debug', type=int, default=0, help='1 = debug mode (analyze one spectrogram only, and output several top candidates). Default = 0.')
    parser.add_argument('-e', '--end', type=str, default='', help='Optional end time in hh:mm:ss format, where hh and mm are optional.')
    parser.add_argument('-i', '--input', type=str, default='', help='Input path (single audio file or directory). No default.')
    parser.add_argument('--date', type=str, default=None, help=f'Date in yyyymmdd, mmdd, or file. Specifying file extracts the date from the file name, using the reg ex defined in config.py.')
    parser.add_argument('--lat', type=float, default=None, help=f'Latitude')
    parser.add_argument('--lon', type=float, default=None, help=f'Longitude')
    parser.add_argument('-o', '--output', type=str, default='', help='Output directory to contain Audacity label files. Default is input directory.')
    parser.add_argument('-p', '--prob', type=float, default=cfg.min_prob, help=f'Minimum confidence level. Default = {cfg.min_prob}.')
    parser.add_argument('-s', '--start', type=str, default='', help='Optional start time in hh:mm:ss format, where hh and mm are optional.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')
    start_time = time.time()
    logging.info('Initializing')

    if args.prob < 0:
        logging.error('Error: prob must be >= 0')
        quit()

    cfg.check_adjacent = (args.adj == 1)
    cfg.use_banding_codes = (args.band == 1)
    cfg.min_prob = args.prob

    analyzer = Analyzer(args.input, args.output, args.start, args.end, args.date, args.lat, args.lon, args.debug)
    analyzer.run(start_time)