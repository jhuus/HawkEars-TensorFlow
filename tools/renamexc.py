# Rename files downloaded from Xeno-Canto by keeping only the XC* prefix.
# For instance, rename "XC164583 - Winter Wren - Troglodytes hiemalis.mp3" to "XC164583.mp3".

import argparse
import inspect
import os
import sys

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from core import constants
from core import util

class Main:
    def __init__(self, path):
        self.path = path
        
    def run(self):
        for file_name in os.listdir(self.path):
            file_path = os.path.join(self.path, file_name) 
            if os.path.isfile(file_path):
                base, ext = os.path.splitext(file_path)
                print(base, ext)
                if 'xc' in base.lower() and ext.lower() == '.mp3':
                    tokens = base.split(' ')
                    if len(tokens) > 1:
                        new_file_name = f'{tokens[0]}{ext.lower()}'
                        new_file_path = os.path.join(self.path, new_file_name) 
                        os.rename(file_path, new_file_path)
        
if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='', help='Path to directory containing downloaded files.')
    args = parser.parse_args()

    Main(args.d).run()