# Analyze an audio file, or all audio files in a directory.
# For each audio file, extract spectrograms, analyze them and output an Audacity label file
# with the class predictions.

import argparse
import os
import sys
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
    def __init__(self, checkpoint_path, input_path, output_path, min_prob, merge, start_time, end_time):
        self.checkpoint_path = checkpoint_path
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.min_prob = min_prob
        self.merge = (merge == 1)
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        
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
        
        self.classes = util.get_file_lines(constants.CLASSES_FILE)
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
        
    # given the predictions for a class, return the predicted class,
    # ignoring classes in the ignore file
    def _get_predicted_class(self, predictions):
        predicted_class = np.argmax(predictions)
        class_name = self.classes[predicted_class]
        if class_name in self.ignore:
            return None, None
        
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

        offsets = self.audio.find_sounds()
        specs = self._get_specs(offsets)
        predictions = self.model.predict(specs)
        
        labels = []
        prev_label = None
        for i in range(len(predictions)):
            predicted_class, class_name = self._get_predicted_class(predictions[i])
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
        output_path = os.path.join(self.output_path, f'{tokens[0]}_BirdEars_Audacity_Labels.txt')
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
        
        for file_path in file_list:
            self._analyze_file(file_path)

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default=constants.CKPT_PATH, help='Checkpoint path. Default=data/ckpt')
    parser.add_argument('-e', type=str, default='', help='Optional end time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-i', type=str, default='', help='Input path (single audio file or directory). No default')
    parser.add_argument('-m', type=int, default=1, help='1 = Merge adjacent labels, 0 = do not. Default = 1')
    parser.add_argument('-o', type=str, default='', help='Output directory to contain Audacity label files. Default is current directory')
    parser.add_argument('-p', type=float, default=0.90, help='Minimum match weight. Default = 0.90')
    parser.add_argument('-s', type=str, default='', help='Optional start time in hh:mm:ss format, where hh and mm are optional')
    args = parser.parse_args()
        
    analyzer = Analyzer(args.c, args.i, args.o, args.p, args.m, args.s, args.e)
    analyzer.run()