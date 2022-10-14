# Use iNaturalist API to find and download recordings for a species.

import argparse
import os
import pyinaturalist
import requests

class Main:
    def __init__(self, species_name, output_path, max_downloads, rename):
        self.species_name = species_name
        self.output_path = output_path
        self.max_downloads = max_downloads
        self.rename = rename
        
        if len(self.species_name) == 0:
            self._fatal_error("Species name must be specified.")
        
        if len(self.output_path) == 0:
            self._fatal_error("Output path must be specified.")

    def _fatal_error(self, message):
        print(message)
        quit()

    def _download(self, url):
        if url is None or len(url.strip()) == 0:
            return

        tokens = url.split('?')
        tokens2 = tokens[0].split('/')
        filename = tokens2[-1]

        base, ext = os.path.splitext(filename)

        # check mp3_path too in case file was converted to mp3
        if self.rename:
            output_path = f'{self.output_path}/N{filename}'
            mp3_path = f'{self.output_path}/N{base}.mp3'

        else:
            output_path = f'{self.output_path}/{filename}'
            mp3_path = f'{self.output_path}/{base}.mp3'

        if not os.path.exists(output_path) and not os.path.exists(mp3_path):
            print(f'Downloading {output_path}')
            r = requests.get(url, allow_redirects=True)
            open(output_path, 'wb').write(r.content)
            self.num_downloads += 1

    def run(self):
        if not os.path.exists(self.output_path):
            try:
                os.makedirs(self.output_path)
            except:
                self._fatal_error(f'Error creating {self.output_path}')

        self.num_downloads = 0
        response = pyinaturalist.get_observations(taxon_name=f'{self.species_name}', sounds=True, per_page=self.max_downloads)
        for result in response['results']:
            for sound in result['sounds']:
                self._download(sound['file_url'])

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, default='', help='Path to output directory.')
    parser.add_argument('-n', type=int, default=500, help='Maximum number of recordings to download. Default = 500.')
    parser.add_argument('-r', type=int, default=1, help='1 = rename by adding an N prefix, 0 = do not rename (default = 1).')
    parser.add_argument('-s', type=str, default='', help='Species name.')

    args = parser.parse_args()

    Main(args.s, args.o, args.n, args.r == 1).run()
            