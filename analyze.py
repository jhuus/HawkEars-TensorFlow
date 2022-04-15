# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and output an Audacity label file
# with the class predictions.

import argparse
import os
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
from core import constants
from core import util

# when checking if adjacent segment matches species, use self.min_prob minus this
SUBTRACT_ADJACENT_PROB = 0.20

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.reset()

    def reset(self):
        self.has_label = False
        self.probs_s = [] # from single-label model predictions (one per segment)
        self.probs_m = [] # from multi-label model predictions (one per segment)

class Label:
    def __init__(self, class_name, probability, start_time, end_time):
        self.class_name = class_name
        self.probability = probability
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, class_file_path, input_path, output_path, min_prob_s, min_prob_m, start_time, 
                 end_time, use_codes, segmentation_mode, use_ignore_file):
                 
        self.ckpt_path_s = 'data/ckpt_s' # single-label model
        self.ckpt_path_m = 'data/ckpt_m' # multi-label model
        self.class_file_path = class_file_path
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.min_prob_s = min_prob_s # for single-label model
        self.min_prob_m = min_prob_m # for multi-label model
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.use_codes = (use_codes == 1)
        self.segmentation_mode = segmentation_mode
        self.use_ignore_file = (use_ignore_file == 1)
        
        if self.start_seconds != None and self.end_seconds != None and self.end_seconds <= self.start_seconds:
            print('Error: end time must be greater than start time')
            sys.exit()
        
        # if no output path specified and input path is a directory,
        # put the output labels in the input directory
        if len(self.output_path) == 0:
            if os.path.isdir(self.input_path):
                self.output_path = self.input_path
        elif not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        self.class_infos = self._get_class_infos()
        self.audio = audio.Audio()

    def _get_class_infos(self):
        classes = util.get_class_list(class_file_path=self.class_file_path)
        class_dict = util.get_class_dict(class_file_path=self.class_file_path)
        ignore_list = util.get_file_lines(constants.IGNORE_FILE)

        class_infos = []
        for klass in classes: # use "klass" since "class" is a keyword
            if self.use_ignore_file and klass in ignore_list:
                ignore = True
            else:
                ignore = False

            class_infos.append(ClassInfo(klass, class_dict[klass], ignore))

        return class_infos

    # return predictions for both the single and multi-label models
    def _get_predictions(self, signal, rate):  
        # if needed, pad the signal with zeros to get the last spectrogram
        total_seconds = len(signal) / rate
        last_segment_len = total_seconds - constants.SEGMENT_LEN * (total_seconds // constants.SEGMENT_LEN)
        if last_segment_len > 0.5:
            # more than 1/2 a second at the end, so we'd better analyze it
            pad_amount = int(rate * (constants.SEGMENT_LEN - last_segment_len)) + 1
            signal = np.pad(signal, (0, pad_amount), 'constant', constant_values=(0, 0))
        
        if self.start_seconds is None:
            start_seconds = 0
        else:
            start_seconds = self.start_seconds

        if self.end_seconds is None:
            # ensure >= 1 so offset 0 is included for very short recordings
            end_seconds = max(1, (len(signal) / rate) - constants.SEGMENT_LEN)
        else:
            end_seconds = self.end_seconds

        if self.segmentation_mode == 0:
            offsets = np.arange(start_seconds, end_seconds, 1.0).tolist()
        else:
            offsets = np.arange(start_seconds, end_seconds, 3.0).tolist()
        
        specs = self._get_specs(offsets)
        predictions_s = self.model_s.predict(specs)

        if self.min_prob_m <= 1:
            predictions_m = self.model_m.predict(specs)
        else:
            predictions_m = None

        return predictions_s, predictions_m, offsets

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

        seconds += int(tokens[-1])
        return seconds
        
    # get a list of spectrograms for the given offsets
    def _get_specs(self, offsets):
        spec_list = self.audio.get_spectrograms(offsets)
        spec_array = np.zeros((len(offsets), constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1))
        for i in range(len(offsets)):
            spec = spec_list[i].reshape((constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1))
            spec_array[i] = spec.astype(np.float32)
            
        return spec_array

    def _analyze_file(self, file_path):
        print(f'Analyzing {file_path}')

        # clear info from previous recording
        for class_info in self.class_infos:
            class_info.reset()

        signal, rate = self.audio.load(file_path)
        if not self.audio.have_signal:
            return

        predictions_s, predictions_m, offsets = self._get_predictions(signal, rate)

        # populate class_infos with predictions
        for i in range(len(predictions_s)):
            for j in range(len(predictions_s[i])):
                self.class_infos[j].probs_s.append(predictions_s[i][j])

                if predictions_m is None:
                    self.class_infos[j].probs_m.append(predictions_s[i][j])
                else:
                    self.class_infos[j].probs_m.append(predictions_m[i][j])

                if ((predictions_s[i][j] >= self.min_prob_s and (predictions_m is None or predictions_m[i][j] >= self.min_prob_s)) or 
                    (predictions_m is not None and predictions_m[i][j] >= self.min_prob_m)):
                    self.class_infos[j].has_label = True

        # generate labels for one class at a time
        labels = []
        for class_info in self.class_infos:
            if class_info.ignore or not class_info.has_label:
                continue

            if self.use_codes:
                name = class_info.code
            else:
                name = class_info.name

            prev_label = None
            probs_s = class_info.probs_s
            probs_m = class_info.probs_m
            min_adj_prob_s = self.min_prob_s - SUBTRACT_ADJACENT_PROB # in mode 0, adjacent segments need this prob at least
            min_adj_prob_m = self.min_prob_m - SUBTRACT_ADJACENT_PROB # in mode 0, adjacent segments need this prob at least
            for i in range(len(probs_s)):
                # create a label if both models exceed min_prob_s (p1) or multi-label model exceeds min_prob_m (p2) 
                if probs_s[i] >= self.min_prob_s and probs_m[i] >= self.min_prob_s:
                    use_multi = False
                    use_prob = probs_s[i]
                elif probs_m[i] >= self.min_prob_m:
                    use_multi = True
                    use_prob = probs_m[i]
                else:
                    continue

                end_time = offsets[i]+constants.SEGMENT_LEN
                if self.segmentation_mode == 0:
                    if i not in [0, len(probs_s) - 1]:
                        if use_multi:
                            if probs_m[i - 1] < min_adj_prob_m and probs_m[i + 1] < min_adj_prob_m:
                                continue
                        elif probs_s[i - 1] < min_adj_prob_s and probs_s[i + 1] < min_adj_prob_s:
                            continue

                    if prev_label != None and prev_label.end_time >= offsets[i]:
                        # extend the previous label's end time (i.e. merge)
                        prev_label.end_time = end_time
                        prev_label.probability = max(use_prob, prev_label.probability)
                    else:
                        label = Label(name, use_prob, offsets[i], end_time)
                        labels.append(label)
                        prev_label = label
                else:
                    label = Label(name, use_prob, offsets[i], end_time)
                    labels.append(label)
                    prev_label = label
                
        self._save_labels(labels, file_path)

    def _save_labels(self, labels, file_path):
        basename = os.path.basename(file_path)
        tokens = basename.split('.')
        output_path = os.path.join(self.output_path, f'{tokens[0]}_HawkEars_Audacity_Labels.txt')
        print(f'Writing output to {output_path}')
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_name};{label.probability:.2f}\n')
                    
        except:
            print(f'Unable to write file {output_path}')
            sys.exit()
        
    def _get_file_list(self):
        if os.path.isdir(self.input_path):
            return util.get_audio_files(self.input_path)
        elif util.is_audio_file(self.input_path):
            return [self.input_path]
        else:
            print(f'{self.input_path} is not a directory or an audio file.')
            sys.exit()
        
    def run(self):
        file_list = self._get_file_list()

        # load the single and multi-label models
        self.model_s = keras.models.load_model(self.ckpt_path_s, compile=False)
        self.model_m = keras.models.load_model(self.ckpt_path_m, compile=False)
        
        start_time = time.time()
        for file_path in file_list:
            self._analyze_file(file_path)
            
        elapsed = time.time() - start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        print(f'Elapsed time for analysis = {minutes}m {seconds}s')

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=0, help='0 = Use species names in labels, 1 = Use banding codes. Default = 0')
    parser.add_argument('-d', type=str, default=constants.CLASSES_FILE, help='Class file path. Default=data/classes.txt')
    parser.add_argument('-e', type=str, default='', help='Optional end time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-i', type=str, default='', help='Input path (single audio file or directory). No default')
    parser.add_argument('-m', type=int, default=0, help='Segmentation mode. 0 = every 1 second (and compare & merge neighbours), 1 = every 3 seconds. Default = 0')
    parser.add_argument('-o', type=str, default='', help='Output directory to contain Audacity label files. Default is current directory')
    parser.add_argument('-p1', type=float, default=0.75, help='Minimum confidence level for single-label model. Default = 0.75')
    parser.add_argument('-p2', type=float, default=0.85, help='Minimum confidence level for multi-label model. Default = 0.85')
    parser.add_argument('-s', type=str, default='', help='Optional start time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-x', type=int, default=1, help='1 = Ignore classes listed in ignore.txt, 0 = do not. Default = 1')
    args = parser.parse_args()

    if args.p1 < 0 or args.p2 < 0:
        print('Error: p1 and p2 must be >= 0')
        quit()

    analyzer = Analyzer(args.d, args.i, args.o, args.p1, args.p2, args.s, args.e, args.b, args.m, args.x)
    analyzer.run()