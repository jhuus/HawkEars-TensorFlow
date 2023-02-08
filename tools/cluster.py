# Spectrogram clustering tool using HDBSCAN (https://hdbscan.readthedocs.io/en/latest/).
# By default it plots one sample spectrogram per cluster to the output directory,
# which defaults to "clusters". Sample output image name is "10-1716-N238913.mp3-21.00.png",
# where 10 is the cluster number, 1716 is the number of spectrograms in the cluster,
# N238913.mp3 is the audio recording containing the spectrogram shown, and 21.00 is the offset
# in the recording corresponding to the spectrogram shown.
# See command line parameters for more details.

import argparse
import inspect
import os
import pickle
import random
import sys
import time

# this is necessary before importing from a peer directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from core import config as cfg
from core import database
from core import util
from core import plot

import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = no info, 2 = no warnings, 3 = no errors

def fatal_error(msg):
    print(msg)
    quit()

class Main:
    def __init__(self, db_path, cluster_num, num_to_plot, output_path, species_name, min_cluster_size, min_samples, center_search_num, gray_scale):
        self.db_path = db_path
        self.cluster_num = cluster_num
        self.num_to_plot = num_to_plot
        self.output_path = output_path
        self.species_name = species_name
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.center_search_num = center_search_num
        self.gray_scale = gray_scale
        self.denoiser = None

    # create clusters of the embeddings;
    # output is a dictionary with a key per cluster and a list of offsets per key
    def _cluster(self):
        cluster_alg = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        cluster_alg.fit(self.embeddings)
        labels = cluster_alg.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = [i]
            else:
                clusters[label].append(i)

        return clusters

    # return a dictionary with cluster numbers as keys and the index of a good example of each cluster as values
    def _find_examples(self, clusters):
        examples = {}
        for label in sorted(clusters.keys()):
            # try center_search_num random examples, and pick the one with the shortest max distance to the rest of the cluster
            best_max_dist = sys.maxsize
            best_index = None
            count = min(self.center_search_num, len(clusters[label]))
            indices = util.get_rand_list(count, len(clusters[label]) - 1)
            for i in range(count):
                max_dist = 0
                idx1 = clusters[label][indices[i]]
                for j in range(len(clusters[label])):
                    if idx1 == j:
                        continue

                    idx2 = clusters[label][j]

                    dist = scipy.spatial.distance.cosine(self.embeddings[idx1], self.embeddings[idx2])
                    max_dist = max(dist, max_dist)

                if max_dist < best_max_dist:
                    best_max_dist = max_dist
                    best_index = idx1

            examples[label] = best_index

        return examples

    # get spectrograms from database
    def _load_spectrograms(self, db_path, species_name):
        db = database.Database(db_path)
        results = db.get_spectrogram_by_subcat_name(species_name, include_embedding=True)

        if results is None or len(results) == 0:
            fatal_error('No spectrograms found.')

        self.embeddings = []
        self.specs = []
        self.spec_names = []
        for r in results:
            self.embeddings.append(np.frombuffer(r.embedding, dtype=np.float16))
            self.specs.append(util.expand_spectrogram(r.value))
            basename, ext = os.path.splitext(r.filename)
            self.spec_names.append(f'{basename}-{r.offset:.2f}')

    # plot the spectrograms in a single cluster;
    # ideally ones with background species or odd noises would be plotted first,
    # with the cleanest ones at the end; that way it's easy to filter
    # out the bad ones and import the rest; so try to do that
    def _plot_cluster(self, clusters, cluster_num):
        curr_folder = os.path.join(self.output_path, f'{cluster_num}')
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)

        if self.num_to_plot is None:
            num_to_plot = len(clusters[cluster_num])
        else:
            num_to_plot = min(self.num_to_plot, len(clusters[cluster_num]))

        # create denoised versions, so simple noise is ignored in sorting
        if self.denoiser is None:
            self.denoiser = keras.models.load_model("../data/denoiser", compile=False)

        specs = np.zeros((num_to_plot, cfg.spec_height, cfg.spec_width, 1))
        for i in range(num_to_plot):
            offset = clusters[cluster_num][i]
            specs[i] = self.specs[offset].reshape((cfg.spec_height, cfg.spec_width, 1))

        denoised_specs = self.denoiser.predict(specs, verbose=0)

        # for each denoised spec, normalize it then count the number of pixels > .02
        counts = []
        for i, spec in enumerate(denoised_specs):
            # normalize the values to [0, 1], since denoising tends to yield about [0, .6]
            max = spec.max()
            if max > 0:
                spec = spec / max

            spec = spec.clip(0, 1) # probably not needed, but just to be safe

            # count pixels > .02
            count = (spec > .02).sum()
            counts.append((i, count))

        # sort by count descending, then plot them in that order
        sorted_counts = sorted(counts, key=lambda value: -value[1])
        for i, count in enumerate(sorted_counts):
            index = count[0]
            offset = clusters[cluster_num][index]
            curr_path = os.path.join(curr_folder, f'{i}~{self.spec_names[offset]}.png')
            if not os.path.exists(curr_path):
                plot.plot_spec(self.specs[offset], curr_path, gray_scale=self.gray_scale)

    def run(self):
        self._load_spectrograms(self.db_path, self.species_name)
        print(f'retrieved {len(self.specs)} spectrograms')

        # typical use is to run this once without the -c option, so only one example is plotted per cluster,
        # and then one or more times with the -c option to plot details per cluster; saving a pickled instance
        # of the cluster dict saves time after the 1st run, so we don't have to rerun the clustering algorithm
        pickle_path = os.path.join(output_path, 'clusters.pkl')
        if os.path.exists(pickle_path):
            pickle_file = open(pickle_path, 'rb')
            clusters = pickle.load(pickle_file)
        else:
            clusters = self._cluster()
            pickle_file = open(pickle_path, 'wb')
            pickle.dump(clusters, pickle_file)

        if cluster_num is None:
            # find and plot one good example of each cluster
            examples = self._find_examples(clusters)
            for key in examples:
                offset = examples[key]
                count = len(clusters[key])
                curr_path = os.path.join(self.output_path, f'{key}~{count}~{self.spec_names[offset]}.png')
                plot.plot_spec(self.specs[offset], curr_path, gray_scale=self.gray_scale)
        else:
            # plot the spectrograms in a specified cluster
            self._plot_cluster(clusters, self.cluster_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=None, help='If specified, plot spectrograms for this cluster. Default = None.')
    parser.add_argument('-d', type=str, default='../data/training.db', help='Database path.')
    parser.add_argument('-g', type=int, default=0, help='1 = gray scale plots, 0 = colour. Default = 0.')
    parser.add_argument('-n', type=int, default=None, help='If -c is specified, optionally plot only this many spectrograms. Default is all.')
    parser.add_argument('-p1', type=int, default=15, help='min_cluster_size parameter to HDBSCAN algorithm. Default = 10.')
    parser.add_argument('-p2', type=int, default=7, help='min_samples parameter to HDBSCAN algorithm. Default = 7.')
    parser.add_argument('-p3', type=int, default=10, help='number of samples to try when finding good representative of each cluster. Default = 10.')
    parser.add_argument('-o', type=str, default='clusters', help='Output directory name.')
    parser.add_argument('-s', type=str, default='', help='Species name.')

    args = parser.parse_args()
    cluster_num = args.c
    db_path = args.d
    gray_scale = args.g == 1
    num_to_plot = args.n
    min_cluster_size = args.p1
    min_samples = args.p2
    center_search_num = args.p3
    output_path = args.o
    species_name = args.s

    if os.path.exists(output_path) and cluster_num is None:
        # unless we're plotting a specific cluster, be sure to start fresh
        os.rmdir(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    start_time = time.time()

    Main(db_path, cluster_num, num_to_plot, output_path, species_name, min_cluster_size, min_samples, center_search_num, gray_scale).run()

    elapsed = time.time() - start_time
    print(f'elapsed seconds = {elapsed:.3f}')
