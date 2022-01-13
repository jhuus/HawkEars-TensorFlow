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

import numpy as np
import tensorflow as tf
from tensorflow import keras

from core import audio
from core import constants
from core import util

class Label:
    def __init__(self, class_name, probability, start_time, end_time):
        self.class_name = class_name
        self.probability = probability
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, checkpoint_path, input_path, output_path, min_prob, merge, start_time, end_time, use_codes, segmentation_mode, 
                 check_shifted, sound_factor, shift_offset, use_ignore_file):
                 
        self.checkpoint_path = checkpoint_path
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.min_prob = min_prob
        self.merge = (merge == 1)
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.use_codes = (use_codes == 1)
        self.segmentation_mode = segmentation_mode
        self.check_shifted = (check_shifted)
        self.sound_factor = sound_factor
        self.shift_offset = shift_offset
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
        
        self.classes = util.get_class_list()
        self.class_dict = util.get_class_dict()
        self.ignore = util.get_file_lines(constants.IGNORE_FILE)
        self.audio = audio.Audio()
        
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
        
    # given the predictions for a class, return the predicted class, ignoring classes in the ignore file;
    # if check_prediction is not None, check if they match
    def _get_predicted_class(self, predictions, check_predictions):
        predicted_class = np.argmax(predictions)
        class_name = self.classes[predicted_class]
        if self.use_ignore_file and class_name in self.ignore:
            return None, None
            
        if check_predictions is not None:
            check_class = np.argmax(check_predictions)
            if predicted_class != check_class:
                # doesn't match a slightly shifted spectrogram, so probably spurious
                return None, None
            
        if self.use_codes:
            return predicted_class, self.class_dict[class_name]
        else:
            return predicted_class, class_name
        
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
        signal, rate = self.audio.load(file_path)
        if not self.audio.have_signal:
            return
            
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
            end_seconds = (len(signal) / rate) - constants.SEGMENT_LEN
        else:
            end_seconds = self.end_seconds

        if self.segmentation_mode == 0:
            offsets = np.arange(start_seconds, end_seconds, 3.0).tolist()
        elif self.segmentation_mode == 1:
            offsets = np.arange(start_seconds, end_seconds, 1.5).tolist()
        elif self.segmentation_mode == 2:
            offsets = self.audio.find_sounds(sound_factor=self.sound_factor)
        
        specs = self._get_specs(offsets)
        predictions = self.model.predict(specs)

        if self.check_shifted:
            shifted_offsets = []
            for offset in offsets:
                shifted_offsets.append(offset + self.shift_offset)
                
            shifted_specs = self._get_specs(shifted_offsets)
            shifted_predictions = self.model.predict(shifted_specs)

        labels = []
        prev_label = None
        for i in range(len(predictions)):
            if self.check_shifted and i < len(shifted_predictions):
                check_predictions = shifted_predictions[i]
            else:
                check_predictions = None
        
            predicted_class, class_name = self._get_predicted_class(predictions[i], check_predictions)
            if predicted_class is None:
                continue
            
            probability = predictions[i][predicted_class]
            if probability >= self.min_prob:

                end_time = offsets[i]+constants.SEGMENT_LEN
                
                if self.merge and prev_label != None and prev_label.class_name == class_name \
                    and prev_label.end_time >= offsets[i]:
                    
                    # extend the previous label's end time, using the higher probability
                    prev_label.end_time = end_time
                    prev_label.probability = max(probability, prev_label.probability)
                else:
                    label = Label(class_name, probability, offsets[i], end_time)
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
        self.model = keras.models.load_model(self.checkpoint_path, compile=False)
        
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
    parser.add_argument('-c', type=str, default=constants.CKPT_PATH, help='Checkpoint path. Default=data/ckpt')
    parser.add_argument('-e', type=str, default='', help='Optional end time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-g', type=int, default=2, help='Segmentation mode. 0 = every 3 seconds, 1 = every 1.5 seconds, 2 = detect sounds and segment heuristically. Default = 2')
    parser.add_argument('-k', type=int, default=0, help='1 = Check against slightly shifted segments, and include only if they match. Default = 0')
    parser.add_argument('-i', type=str, default='', help='Input path (single audio file or directory). No default')
    parser.add_argument('-m', type=int, default=0, help='1 = Merge adjacent labels, 0 = do not. Default = 0')
    parser.add_argument('-o', type=str, default='', help='Output directory to contain Audacity label files. Default is current directory')
    parser.add_argument('-p', type=float, default=0.95, help='Minimum match weight. Default = 0.95')
    parser.add_argument('-p2', type=float, default=1.2, help='Sound factor when finding sounds with g = 0. Default = 1.2')
    parser.add_argument('-p3', type=float, default=0.3, help='Seconds offset for shifted spectrograms with g = 1. Default = 0.3')
    parser.add_argument('-s', type=str, default='', help='Optional start time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-x', type=int, default=1, help='1 = Ignore classes listed in ignore.txt, 0 = do not. Default = 1')
    args = parser.parse_args()
        
    analyzer = Analyzer(args.c, args.i, args.o, args.p, args.m, args.s, args.e, args.b, args.g, args.k, args.p2, args.p3, args.x)
    analyzer.run()