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
from core import plot
from core import util

TOP_N = 6 # number of top matches to log in debug mode
ADJACENT_PROB_FACTOR = 0.65 # when checking if adjacent segment matches species, use self.min_prob times this
DENOISED_PROB_FACTOR = 0.55 # check denoised prediction if std one is >= min_prob times this

class ClassInfo:
    def __init__(self, name, code, ignore):
        self.name = name
        self.code = code
        self.ignore = ignore
        self.reset()

    def reset(self):
        self.has_label = False
        self.probs = [] # predictions (one per segment)

class Label:
    def __init__(self, class_name, probability, start_time, end_time):
        self.class_name = class_name
        self.probability = probability
        self.start_time = start_time
        self.end_time = end_time

class Analyzer:
    def __init__(self, class_file_path, input_path, output_path, min_prob, start_time, 
                 end_time, use_codes, use_ignore_file, debug_mode, check_adjacent):
                 
        self.ckpt_path = 'data/ckpt'
        self.class_file_path = class_file_path
        self.input_path = input_path.strip()
        self.output_path = output_path.strip()
        self.min_prob = min_prob
        self.start_seconds = self._get_seconds_from_time_string(start_time)
        self.end_seconds = self._get_seconds_from_time_string(end_time)
        self.use_codes = (use_codes == 1)
        self.use_ignore_file = (use_ignore_file == 1)
        self.debug_mode = (debug_mode == 1)
        self.check_adjacent = (check_adjacent == 1)
        
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
        
    def _get_file_list(self):
        if os.path.isdir(self.input_path):
            return util.get_audio_files(self.input_path)
        elif util.is_audio_file(self.input_path):
            return [self.input_path]
        else:
            print(f'{self.input_path} is not a directory or an audio file.')
            sys.exit()

    def _get_prediction(self, prediction, denoised_prediction):
        # use the standard prediction if it is big enough or small enough
        if prediction >= self.min_prob or prediction < self.min_prob * DENOISED_PROB_FACTOR:
            return prediction

        # the standard prediction is close but not quite high enough, so factor in the denoised one
        return max(prediction, denoised_prediction)

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

        if self.debug_mode:
            end_seconds = start_seconds + constants.SEGMENT_LEN # just do one segment in debug mode
        elif self.end_seconds is None:
            # ensure >= 1 so offset 0 is included for very short recordings
            end_seconds = max(1, (len(signal) / rate) - constants.SEGMENT_LEN)
        else:
            end_seconds = self.end_seconds
        
        specs = self._get_specs(start_seconds, end_seconds)

        # get a second set of specs with noise removed
        denoiser = keras.models.load_model("data/denoiser", compile=False)
        denoised_temp = denoiser.predict(specs)
        denoised_specs = np.zeros((len(self.offsets), constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1))
        for i in range(len(self.offsets)):
            spec = denoised_temp[i] / denoised_temp[i].max() # make the max 1
            denoised_specs[i] = np.clip(spec, 0, 1) # clip negative values

        predictions = self.model.predict(specs)
        denoised_predictions = self.model.predict(denoised_specs)

        if self.debug_mode:
            self._log_predictions(predictions, denoised_predictions)

        # populate class_infos with predictions
        for i in range(len(self.offsets)):
            for j in range(len(self.class_infos)):
                    value = self._get_prediction(predictions[i][j], denoised_predictions[i][j])
                    self.class_infos[j].probs.append(value)
                    if (self.class_infos[j].probs[-1] >= self.min_prob):
                        self.class_infos[j].has_label = True

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

    # get the list of spectrograms;
    # for performance, call get_spectrograms on non-overlapping offsets,
    # then create overlapping ones from those
    def _get_specs(self, start_seconds, end_seconds):
        offsets = np.arange(start_seconds, end_seconds, constants.SEGMENT_LEN).tolist()
        raw_specs = self.audio.get_spectrograms(offsets, check_noise=False, update_levels=False)

        specs = []
        sec_width = int(constants.SPEC_WIDTH / constants.SEGMENT_LEN) # width of one second of spectrogram
        self.offsets = np.arange(start_seconds, end_seconds, 1.0).tolist()
        for i in range(len(self.offsets)):
            src_idx = int(i / constants.SEGMENT_LEN)
            mod = i % constants.SEGMENT_LEN
            if mod == 0:
                # no overlap
                spec = raw_specs[src_idx]
            elif src_idx < len(offsets) - 1:
                # concatenate parts of two spectrograms to create an overlapping one
                spec = np.zeros((constants.SPEC_HEIGHT, constants.SPEC_WIDTH))
                offset = (constants.SEGMENT_LEN - mod) * sec_width
                spec[:, :offset] = raw_specs[src_idx][:, mod * sec_width:]
                spec[:, offset:] = raw_specs[src_idx + 1][:, :mod * sec_width]

            specs.append(spec)

        # check low noise and update levels now that overlapping spectrograms have been created
        self.audio.dampen_low_noise(specs)
        self.audio.update_levels(specs)

        for i in range(len(specs)):
            # normalize values between 0 and 1
            max = specs[i].max()
            if max > 0:
                specs[i] /= max
                
            specs[i] = specs[i].clip(0, 1)

        spec_array = np.zeros((len(specs), constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1))
        for i in range(len(specs)):
            spec_array[i] = specs[i].reshape((constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1)).astype(np.float32)

        return spec_array

    def _analyze_file(self, file_path):
        print(f'Analyzing {file_path}')

        # clear info from previous recording
        for class_info in self.class_infos:
            class_info.reset()

        signal, rate = self.audio.load(file_path)

        if not self.audio.have_signal:
            return

        self._get_predictions(signal, rate)

        # generate labels for one class at a time
        labels = []
        min_adj_prob = self.min_prob * ADJACENT_PROB_FACTOR # in mode 0, adjacent segments need this prob at least

        for class_info in self.class_infos:
            if class_info.ignore or not class_info.has_label:
                continue

            if self.use_codes:
                name = class_info.code
            else:
                name = class_info.name

            prev_label = None
            probs = class_info.probs
            for i in range(len(probs)):

                # create a label if probability exceeds the threshold 
                if probs[i] >= self.min_prob:
                    use_prob = probs[i]
                else:
                    continue

                end_time = self.offsets[i]+constants.SEGMENT_LEN
                if self.check_adjacent:
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
        print(f'Writing output to {output_path}')
        try:
            with open(output_path, 'w') as file:
                for label in labels:
                    file.write(f'{label.start_time:.2f}\t{label.end_time:.2f}\t{label.class_name};{label.probability:.2f}\n')
                    
        except:
            print(f'Unable to write file {output_path}')
            sys.exit()

    # in debug mode, output the top predictions
    def _log_predictions(self, predictions, denoised_predictions):
        predictions = np.copy(predictions[0])
        print("\ntop predictions")

        for i in range(TOP_N):
            j = np.argmax(predictions)
            code = self.class_infos[j].code
            confidence = predictions[j]
            print(f"{code}: {confidence}")
            predictions[j] = 0

        print("\ntop predictions (denoised)")

        denoised_predictions = np.copy(denoised_predictions[0])
        for i in range(TOP_N):
            j = np.argmax(denoised_predictions)
            code = self.class_infos[j].code
            confidence = denoised_predictions[j]
            print(f"{code}: {confidence}")
            denoised_predictions[j] = 0

        print("")

    def run(self):
        file_list = self._get_file_list()

        # load the model
        self.model = keras.models.load_model(self.ckpt_path, compile=False)
        
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
    parser.add_argument('-a', type=int, default=1, help='1 = Ignore if neither adjacent/overlapping spectrogram matches. Default = 1')
    parser.add_argument('-b', type=int, default=0, help='0 = Use species names in labels, 1 = Use banding codes. Default = 0')
    parser.add_argument('-d', type=str, default=constants.CLASSES_FILE, help='Class file path. Default=data/classes.txt')
    parser.add_argument('-e', type=str, default='', help='Optional end time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-g', type=int, default=0, help='1 = debug mode (analyze one spectrogram only, and output several top candidates). Default = 0')
    parser.add_argument('-i', type=str, default='', help='Input path (single audio file or directory). No default')
    parser.add_argument('-o', type=str, default='', help='Output directory to contain Audacity label files. Default is current directory')
    parser.add_argument('-p', type=float, default=0.9, help='Minimum confidence level. Default = 0.9')
    parser.add_argument('-s', type=str, default='', help='Optional start time in hh:mm:ss format, where hh and mm are optional')
    parser.add_argument('-x', type=int, default=1, help='1 = Ignore classes listed in ignore.txt, 0 = do not. Default = 1')
    args = parser.parse_args()

    if args.p < 0:
        print('Error: p must be >= 0')
        quit()

    analyzer = Analyzer(args.d, args.i, args.o, args.p, args.s, args.e, args.b, args.x, args.g, args.a)
    analyzer.run()