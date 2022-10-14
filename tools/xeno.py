# Download files for a given species from Xeno-Canto.

import argparse
import json
import os
import requests
import shutil
import sys
import time
import zlib
            
# given a time string in hh:mm:ss format, return seconds
def extract_seconds(timeStr):
    tokens = timeStr.split(':')
    ms = 0
    if len(tokens) > 0:
        ms += int(tokens[-1])
    
    if len(tokens) > 1:
        ms += int(tokens[-2]) * 60
    
    if len(tokens) > 2:
        ms += int(tokens[-3]) * 3600
    
    return ms

# sort recordings by quality
# if no quality given, sort between A and B
def sort_key(recording):
    quality = recording['q']
    
    if quality == 'A':
        return 1
    elif quality == '':
        return 2
    elif quality == 'B':
        return 3
    elif quality == 'C':
        return 4
    elif quality == 'D':
        return 5
    else:
        return 6

class Main:
    def __init__(self, species_name, output_dir, max_downloads, min_seconds, max_seconds):
        self.species_name = species_name
        self.output_dir = output_dir
        self.max_downloads = max_downloads
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    def _get_recordings_list(self):
        self.recordings = []
        page = 0
        done = False
        while not done:
            page += 1
            encoded = self.species_name.lower().replace(' ', '+')
            url = f'https://www.xeno-canto.org/api/2/recordings?query={encoded}&page={page}'
            print(f'Requesting data from {url}')
            response = requests.get(url)
            if response.status_code == 200:
                done = self._process_response(response.text)
            else:
                print(f'HTTP GET failed with status={response.status_code}')
                done = True
    
    def _process_response(self, text):
        j = json.loads(text)
        page = j['page']
        num_pages = j['numPages']
        recordings = j['recordings']
        
        print(f'Response contains {len(recordings)} recordings.')
        for recording in recordings:
            self.recordings.append(recording)
        
        if page == num_pages:
            return True  # done
        else:
            return False # not done

    def _download_recordings(self):
        downloaded = 1
        for recording in self.recordings:
            outfile = os.path.join(self.output_dir, f"XC{recording['id']}.mp3")
            if not os.path.exists(outfile):
                print(f'Downloading {outfile}')
                url = recording['file']
                response = requests.get(url)
                with open(outfile, 'wb') as mp3:
                    mp3.write(response.content)
                    downloaded += 1

                    if self.max_downloads > 0 and downloaded >= self.max_downloads:
                        return

    def run(self):
        # get list of recordings and remove any that are too short or too long
        self._get_recordings_list()
        remove = []
        for recording in self.recordings:
            seconds = extract_seconds(recording['length'])

            if self.min_seconds > 0 and seconds < self.min_seconds:
                remove.append(recording)
            elif self.max_seconds > 0 and seconds > self.max_seconds:
                remove.append(recording)
            
        for recording in remove:
            self.recordings.remove(recording)

        # sort the list, then start downloading
        self.recordings.sort(key=sort_key)
        self._download_recordings()

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='', help='Species name.')
    parser.add_argument('-o', type=str, default='', help='Path to output directory.')
    parser.add_argument('-n', type=int, default=0, help='Maximum number of recordings to download. Default = 0, i.e. all available.')
    parser.add_argument('-m1', type=int, default=0, help='Minimum recording length in seconds. Default = 0, i.e. no minimum.')
    parser.add_argument('-m2', type=int, default=0, help='Maximum recording length in seconds. Default = 0, i.e. no maximum.')

    args = parser.parse_args()

    Main(args.s, args.o, args.n, args.m1, args.m2).run()
            