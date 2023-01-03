# Train a denoiser.

import argparse
import math
import os
import random
import time

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
from model import efficientnet_v2_autoencoder

BASE_LR = .0008
BATCH_SIZE = 32
CACHE_LEN = 1000 # cache this many noise specs for performance
NOISE_VARIANCE = 0.0015 # larger variances lead to more noise
NOISE_MULT_MIN = 0.2 # when adding noise to image, multiply it by rand(NOISE_MULT_MIN, NOISE_MULT_MAX)
NOISE_MULT_MAX = 0.6

NUM_EPOCHS = 10
SAVE_EPOCH = 10 # save a ckpt starting at this epoch

#NUM_EPOCHS = 7
#SAVE_EPOCH = 7 # save a ckpt starting at this epoch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

class DataGenerator():
    def __init__(self, db, x_train, low_noise_detector):
        self.audio = audio.Audio()
        self.x_train = x_train
        self.indices = np.arange(len(x_train))

        # get noise spectrograms from the database
        results = db.get_spectrogram_by_subcat_name('Noise')
        print(f'Fetched {len(results)} noise spectrograms from the database.')

        spec_height = cfg.lnd_spec_height if low_noise_detector else cfg.spec_height
        self.real_noise = np.zeros((len(results), spec_height, cfg.spec_width, 1))
        for i, r in enumerate(results):
            self.real_noise[i] = util.expand_spectrogram(r.value, low_noise_detector=low_noise_detector)

        # create some artificial noise
        self.white_noise = np.zeros((CACHE_LEN, spec_height, cfg.spec_width, 1))
        for i in range(CACHE_LEN):
            self.white_noise[i] = skimage.util.random_noise(self.white_noise[i], mode='gaussian', var=NOISE_VARIANCE, clip=True)
            self.white_noise[i] /= np.max(self.white_noise[i]) # scale so max value is 1

        self.pink_noise = np.zeros((CACHE_LEN, spec_height, cfg.spec_width, 1))
        for i in range(CACHE_LEN):
            self.pink_noise[i] = self.audio.pink_noise()[:spec_height, :, :]

    # this is called once per epoch to generate the spectrograms
    def __call__(self):
        np.random.shuffle(self.indices)
        for i, id in enumerate(self.indices):
            spec = self.x_train[id]
            r = random.uniform(0, 1)
            if r < 0.05:
                noisy_spec = self.add_noise(spec, self.white_noise)
            elif r < 0.1:
                noisy_spec = self.add_noise(spec, self.pink_noise)
            else:
                noisy_spec = self.add_noise(spec, self.real_noise)

            yield (noisy_spec.astype(np.float32), spec.astype(np.float32))

    # return a copy of the spec with noise added
    def add_noise(self, spec, noise):
        noisy_spec = np.copy(spec)
        index = random.randint(0, len(noise) - 1)
        noisy_spec += noise[index] * random.uniform(NOISE_MULT_MIN, NOISE_MULT_MAX)
        noisy_spec /= np.max(noisy_spec)
        noisy_spec = noisy_spec.clip(0, 1)

        return noisy_spec

# learning rate schedule with cosine decay
def cos_lr_schedule(epoch):
    base_lr = BASE_LR * BATCH_SIZE / 64
    return base_lr * (1 + math.cos(epoch * math.pi / max(NUM_EPOCHS, 1))) / 2

# return spectrograms for the given class
# for non-noise classes, remove some with the least or most pixels
def get_spectrograms(db, name, filter):
    if name is None:
        # special case - return 50 empty spectrograms for low noise detector
        specs = []
        for i in range(50):
            specs.append(np.zeros((cfg.lnd_spec_height, cfg.spec_width, 1)))

        return specs

    results = db.get_spectrogram_by_subcat_name(name)
    spec_list = []
    for r in results:
        curr = util.expand_spectrogram(r.value, low_noise_detector=low_noise_detector)
        spec_list.append((curr, len(curr[curr > 0.05])))

    if filter:
        sorted_specs = sorted(spec_list, key=lambda value: value[1])
        filtered_specs = []

        start = int(.05 * len(sorted_specs)) # trim 5% from the cleanest, in case they're nearly empty
        end = int(.75 * len(sorted_specs)) # trim 25% with the most pixels louder than the cutoff
        for i in range(start, end):
            filtered_specs.append(sorted_specs[i][0])
        return filtered_specs
    else:
        specs = []
        for spec in spec_list:
            specs.append(spec[0])

        return specs

# main entry point
parser = argparse.ArgumentParser()
parser.add_argument('-lnd', type=int, default=0, help='1 = low noise detector. Default = 0.')
args = parser.parse_args()
low_noise_detector = (args.lnd == 1)

start_time = time.time()

if low_noise_detector:
    filter = False
    spec_height = cfg.lnd_spec_height
    db = database.Database(filename="data/low_noise.db")
    classes = ["Other", None]
else:
    filter = True
    spec_height = cfg.spec_height
    db = database.Database(filename="data/training.db")
    classes = ["American Crow", "American Redstart", "Chipping Sparrow", "Common Loon", "Common Yellowthroat",
               "Great Horned Owl", "Tufted Titmouse", "Yellow Warbler"]

# define and compile the model
#model = efficientnet_v2_autoencoder.EfficientV2Autoencoder(input_shape=(128, 384, 1), model_type='auto-2')
model = mirnet.mirnet_model(input_shape=[spec_height, cfg.spec_width, 1])
opt = keras.optimizers.Adam(learning_rate = cos_lr_schedule(0))
model.compile(optimizer=opt, loss=keras.losses.MeanAbsoluteError())
model.build(input_shape=(None, 128, 384, 1)) # required before calling summary

'''
# output model summaries, if using EfficientNetV2
dir = 'summary'
if not os.path.exists(dir):
    os.makedirs(dir)

with open(f'{dir}/encoder.txt','w') as text_output:
    model.encoder.summary(print_fn=lambda x: text_output.write(x + '\n'))

with open(f'{dir}/decoder.txt','w') as text_output:
    model.decoder.summary(print_fn=lambda x: text_output.write(x + '\n'))
'''

# create the training dataset
specs = []
for name in classes:
    curr_specs = get_spectrograms(db, name, filter)
    specs += curr_specs

print(f'Fetched {len(specs)} bird spectrograms from the database.')

x_train = [0 for i in range(len(specs))]
train_index =  0
for i in range(len(specs)):
    x_train[train_index] = specs[i]
    train_index += 1

datagen = DataGenerator(db, x_train, low_noise_detector)
train_ds = tf.data.Dataset.from_generator(
    datagen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([spec_height, cfg.spec_width, 1],[spec_height, cfg.spec_width, 1]))
train_ds = train_ds.batch(BATCH_SIZE)

checkpoint_path = "data/denoiser"
if SAVE_EPOCH < NUM_EPOCHS:
    copy_ckpt = True
else:
    copy_ckpt = False

model_checkpoint_callback = model_checkpoint.ModelCheckpoint(checkpoint_path, SAVE_EPOCH, 0, copy_ckpt=copy_ckpt, save_best_only=False)

lr_scheduler = keras.callbacks.LearningRateScheduler(cos_lr_schedule)
callbacks = [model_checkpoint_callback, lr_scheduler]
model.fit(train_ds, epochs=NUM_EPOCHS, validation_data=None, shuffle=True, callbacks=callbacks)

elapsed = time.time() - start_time
minutes = int(elapsed) // 60
seconds = int(elapsed) % 60
print(f'Elapsed time for training = {minutes}m {seconds}s\n')
