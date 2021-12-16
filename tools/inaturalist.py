# Download audio files from iNaturalist, using a list exported to CSV from https://www.inaturalist.org/observations/export.
# Columns should be as follows: id,license,url,sound_url,common_name,taxon_id

import argparse
import os
import csv
import requests
import sys

class Main:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def _fatal_error(self, message):
        print(message)
        sys.exit()

    def _download(self, url):
        print(f'Downloading {url} to {self.output_path}')
        tokens = url.split('?')
        tokens2 = tokens[0].split('/')
        output_path = f'{self.output_path}/{tokens2[-1]}'
        
        r = requests.get(url, allow_redirects=True)
        open(output_path, 'wb').write(r.content)

    def run(self):
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
            except:
                self._fatal_error(f'Error creating {self.output_path}')        

        try:
            with open(self.input_path, mode='r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                url_dict = {}
                for row in csv_reader:
                    if line_count > 0:
                        url = row[3].strip()
                        if len(url) > 0 and not url in url_dict:
                            # not a duplicate
                            self._download(row[3])
                            url_dict[url] = 1
                        
                    line_count += 1
        except:
            self._fatal_error(f'Error processing {self.input_path}')

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='', help='Path to input file.')
    parser.add_argument('-o', type=str, default='', help='Path to output directory.')
    args = parser.parse_args()

    Main(args.i, args.o).run()            
            