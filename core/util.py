# Utility functions

import os
import zlib

import numpy as np
import matplotlib.pyplot as plt

from core import constants

AUDIO_EXTS = [
  '.3gp', '.3gpp', '.8svx', '.aa', '.aac', '.aax', '.act', '.aiff', '.alac', '.amr', '.ape', '.au', 
  '.awb', '.cda', '.dss', '.dvf', '.flac', '.gsm', '.iklax', '.ivs', '.m4a', '.m4b', '.m4p', '.mmf', 
  '.mp3', '.mpc', '.mpga', '.msv', '.nmf', '.octet-stream', '.ogg', '.oga', '.mogg', '.opus', '.org', 
  '.ra', '.rm', '.raw', '.rf64', '.sln', '.tta', '.voc', '.vox', '.wav', '.wma', '.wv', '.webm', '.x-m4a',
]

# compress a spectrogram in preparation for inserting into database
def compress_spectrogram(data):
    data = data * 255
    np_bytes = data.astype(np.uint8)
    bytes = np_bytes.tobytes()
    compressed = zlib.compress(bytes)
    return compressed

# decompress a spectrogram, then convert from bytes to floats and reshape it
def expand_spectrogram(spec, binary_classifier=False):
    bytes = zlib.decompress(spec)
    spec = np.frombuffer(bytes, dtype=np.uint8) / 255
    spec = spec.astype(np.float32)
    
    if binary_classifier:
        spec = spec.reshape(constants.BINARY_SPEC_HEIGHT, constants.SPEC_WIDTH, 1)
    else:
        spec = spec.reshape(constants.SPEC_HEIGHT, constants.SPEC_WIDTH, 1)
    
    return spec

# return list of audio files in the given directory;
# returned file names are fully qualified paths
def get_audio_files(path):
    files = []
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name) 
            if os.path.isfile(file_path):
                base, ext = os.path.splitext(file_path)
                if ext != None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
                    files.append(file_path)
    
    return files
    
# return list of strings representing the lines in a text file,
# removing leading and trailing whitespace and ignoring blank lines 
# and lines that start with #
def get_file_lines(path):
    try:
        with open(path, 'r') as file:
            lines = []
            for line in file.readlines():
                line = line.strip()
                if len(line) > 0 and line[0] != '#':
                    lines.append(line)
            
            return lines
    except IOError:
        print(f'Unable to open input file {path}')
        return []


# return True iff given path is an audio file
def is_audio_file(file_path):
    if os.path.isfile(file_path):
        base, ext = os.path.splitext(file_path)
        if ext != None and len(ext) > 0 and ext.lower() in AUDIO_EXTS:
            return True
    
    return False
    
# save a plot of a spectrogram
def plot_spec(spec, path, binary_classifier=False):
    if spec.ndim == 3:
        if binary_classifier:
            spec = spec.reshape((constants.BINARY_SPEC_HEIGHT, constants.SPEC_WIDTH))
        else:
            spec = spec.reshape((constants.SPEC_HEIGHT, constants.SPEC_WIDTH))

    plt.clf() # clear any existing plot data
    plt.pcolormesh(spec, shading='gouraud')
    plt.savefig(path)
    
