# Train a denoiser.

import inspect
import math
import os
import random
import sys
import time

import colorednoise as cn
import cv2
import numpy as np
import skimage
import tensorflow as tf
from tensorflow import keras

from core import audio
from core import config as cfg
from core import database
from core import plot
from core import util

from model import mirnet
from model import model_checkpoint

BASE_LR = .0008
BATCH_SIZE = 32
CACHE_LEN = 1000 # cache this many noise specs for performance
NOISE_VARIANCE = 0.0015 # larger variances lead to more noise
NUM_EPOCHS = 6
SAVE_EPOCH = 6 # save a ckpt starting at this epoch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

class DataGenerator():
    def __init__(self, db, x_train):
        self.audio = audio.Audio()
        self.x_train = x_train
        self.indices = np.arange(len(x_train))

        # get some noise spectrograms from the database
        results = db.get_spectrograms_by_name('Denoiser')
        self.real_noise = np.zeros((len(results), cfg.spec_height, cfg.spec_width, 1))
        for i in range(len(self.real_noise)):
            self.real_noise[i] = util.expand_spectrogram(results[i][0])

        # create some artificial noise
        self.noise = np.zeros((CACHE_LEN, cfg.spec_height, cfg.spec_width, 1))
        for i in range(CACHE_LEN):
            self.noise[i] = skimage.util.random_noise(self.noise[i], mode='gaussian', var=NOISE_VARIANCE, clip=True)

        self.low_noise = np.zeros((CACHE_LEN, cfg.spec_height, cfg.spec_width, 1))
        for i in range(CACHE_LEN):
            self.low_noise[i] = self.audio.pink_noise() + self.noise[i]

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = self.x_train[id]
            if random.uniform(0, 1) < 0.4:
                spec, noisy_spec = self.add_noise(spec, self.low_noise)
            else:
                spec, noisy_spec = self.add_noise(spec, self.real_noise)

            yield (noisy_spec.astype(np.float32), spec.astype(np.float32))

    # return spec with a little noise added and noisy_spec with a lot of noise added
    def add_noise(self, spec, noise):
        noisy_spec = np.copy(spec)
        index = random.randint(0, CACHE_LEN - 1)

        noisy_spec += noise[index]
        noisy_spec /= np.max(noisy_spec)
        noisy_spec = noisy_spec.clip(0, 1)

        spec += noise[index] * random.uniform(0, 0.3)
        spec /= np.max(spec)
        spec = spec.clip(0, 1)

        return spec, noisy_spec

# heuristic noise removal so clean spectrograms are extra clean;
# for each frequency, subtract a multiple of the average amplitude
def remove_noise(spec, row_factor=0.8):
    if row_factor > 0:
        num_freqs = spec.shape[0]
        for i in range(num_freqs):
            spec[i] -= row_factor * np.average(spec[i])

    return spec

# learning rate schedule with cosine decay
def cos_lr_schedule(epoch):
    base_lr = BASE_LR * BATCH_SIZE / 64
    return base_lr * (1 + math.cos(epoch * math.pi / max(NUM_EPOCHS, 1))) / 2

# return spectrograms for the given class, filtered to remove some with the most pixels,
# which are likely noisy
def get_spectrograms(db, name):
    specs = db.get_spectrograms_by_name(name)
    spec_list = []
    for spec in specs:
        curr = util.expand_spectrogram(spec[0])
        curr = remove_noise(curr)
        spec_list.append((curr, len(curr[curr > 0.05])))

    sorted_specs = sorted(spec_list, key=lambda value: value[1])
    filtered_specs = []

    start = int(.05 * len(sorted_specs)) # trim 5% from the cleanest, in case they're nearly empty
    end = int(.75 * len(sorted_specs)) # trim 25% with the most pixels louder than the cutoff
    for i in range(start, end):
        filtered_specs.append(sorted_specs[i][0])
    return filtered_specs

# main entry point
start_time = time.time()

# mixed precision doesn't work well here
# keras.mixed_precision.set_global_policy("mixed_float16")

# define and compile the model
model = mirnet.mirnet_model(input_shape=[cfg.spec_height, cfg.spec_width, 1])
opt = keras.optimizers.Adam(learning_rate = cos_lr_schedule(0))
model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError())

# create the training dataset
classes = ["American Crow", "American Redstart", "Chipping Sparrow", "Common Loon", "Common Yellowthroat",
           "Great Horned Owl", "Tufted Titmouse", "Yellow Warbler"]
db = database.Database(filename="data/training.db")
specs = []
for name in classes:
    curr_specs = get_spectrograms(db, name)
    specs += curr_specs

x_train = [0 for i in range(len(specs))]
train_index =  0
for i in range(len(specs)):
    x_train[train_index] = specs[i]
    train_index += 1

datagen = DataGenerator(db, x_train)
train_ds = tf.data.Dataset.from_generator(
    datagen,
    output_types=(tf.float16, tf.float16),
    output_shapes=([cfg.spec_height, cfg.spec_width, 1],[cfg.spec_height, cfg.spec_width, 1]))
train_ds = train_ds.batch(BATCH_SIZE)

checkpoint_path = "data/denoiser"
model_checkpoint_callback = model_checkpoint.ModelCheckpoint(checkpoint_path, SAVE_EPOCH, 0, copy_ckpt=True, save_best_only=False)

lr_scheduler = keras.callbacks.LearningRateScheduler(cos_lr_schedule)
callbacks = [model_checkpoint_callback, lr_scheduler]
model.fit(train_ds, epochs=NUM_EPOCHS, validation_data=None, shuffle=True, callbacks=callbacks)

elapsed = time.time() - start_time
minutes = int(elapsed) // 60
seconds = int(elapsed) % 60
print(f'Elapsed time for training = {minutes}m {seconds}s\n')
