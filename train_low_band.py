# Train a neural network to detect Ruffed Grouse drumming, using low frequency audio only.
# To see command-line arguments, run the script with -h argument.

import argparse
import logging
import math
import os
import random
import time
from collections import namedtuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 1 = no info, 2 = no warnings, 3 = no errors

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from core import config as cfg
from core import data_generator
from core import database
from core import util

from model import model_checkpoint
from model import resnest

class Trainer:
    def __init__(self):
        self.db = database.Database(f'data/{cfg.training_db}.db')
        self.classes = ['Drumming', 'NotDrumming']
        self.init()

    # create a plot and save it to the output directory
    def plot_results(self, dir, history, key1, key2 = None):
        plt.clf() # clear any existing plot data
        plt.plot(history.history[key1])

        if key2 != None:
            plt.plot(history.history[key2])

        plt.title(key1)
        plt.ylabel(key1)
        plt.xlabel('epoch')

        if key2 is None:
            plt.legend(['train'], loc='upper left')
        else:
            plt.legend(['train', 'test'], loc='upper left')

        plt.savefig(f'{dir}/{key1}.png')

    # run training
    def run(self):
        start_time = time.time()

        if cfg.seed is not None:
            keras.utils.set_random_seed(cfg.seed)

        history = self.model.fit(self.train_ds, epochs = cfg.num_epochs, verbose = cfg.verbosity, validation_data = None,
            shuffle = False, callbacks = self.callbacks, class_weight = self.get_class_weight())

        elapsed = time.time() - start_time
        minutes = int(elapsed) // 60
        seconds = int(elapsed) % 60
        logging.info(f'Elapsed time for training = {minutes}m {seconds}s\n')

        if cfg.verbosity > 0:
            with open(f'{self.out_dir}/summary.txt','w') as text_output:
                text_output.write(f'EfficientNetV2 config: {cfg.eff_config}\n')
                text_output.write(f'Batch size: {cfg.batch_size}\n')
                text_output.write(f'Epochs: {cfg.num_epochs}\n')

                if 'loss' in history.history:
                    text_output.write(f"Training loss: {history.history['loss'][-1]:.3f}\n")

                if not cfg.multi_label and 'accuracy' in history.history:
                    # output accuracy graph
                    training_accuracy = history.history['accuracy'][-1]
                    text_output.write(f'Training accuracy: {training_accuracy:.3f}\n')
                    self.plot_results(self.out_dir, history, 'accuracy')

                text_output.write(f'Elapsed time for training = {minutes}m {seconds}s\n')

            if cfg.verbosity >= 1 and 'loss' in history.history:
                # output loss graph
                self.plot_results(self.out_dir, history, 'loss')

        return self.model

    # calculate and return class weights
    def get_class_weight(self):
        logging.info('Calculate class weights')
        sum = 0
        for class_name in self.classes:
            sum += self.num_specs[class_name]

        class_weight = {}
        average = sum / len(self.classes)
        avg_sqrt = math.sqrt(average)
        for i, class_name in enumerate(self.classes):
            weight = avg_sqrt / math.sqrt(self.num_specs[class_name])

            logging.info(f'Applying weight {weight:.2f} to {class_name}')

            class_weight[i] = weight

        return class_weight

    def create_model(self):
        logging.info('Create model')
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            class_act = 'softmax' # single-label classifier
            self.model = resnest.ResNest(
                    num_classes=len(self.classes),
                    input_shape=(cfg.low_band_spec_height, cfg.spec_width, 1),
                    num_stages=1,
                    blocks_set=[3],
                    seed=1
            ).build_model(class_act)

            opt = keras.optimizers.Adam(learning_rate = cos_lr_schedule(0))
            if cfg.multi_label:
                loss = keras.losses.BinaryCrossentropy(label_smoothing = cfg.label_smoothing)
                self.model.compile(loss=loss, optimizer=opt)
            else:
                loss = keras.losses.CategoricalCrossentropy(label_smoothing = cfg.label_smoothing)
                self.model.compile(loss=loss, optimizer=opt, metrics='accuracy')

    # initialize
    def init(self):
        if cfg.seed is not None:
            keras.utils.set_random_seed(cfg.seed)

        # get the counts per class
        train_total = 0
        num_spectrograms = []
        for i in range(len(self.classes)):
            count = self.db.get_spectrogram_count(self.classes[i])
            num_spectrograms.append(count)
            train_total += count

        logging.info(f'# training samples: {train_total}')

        # initialize arrays
        self.x_train = [0 for i in range(train_total)]
        self.y_train = np.zeros((train_total, len(self.classes)))
        self.train_class = ['' for i in range(train_total)]
        self.input_shape = (cfg.low_band_spec_height, cfg.spec_width, 1)

        # populate from the database;
        # they will be selected randomly per mini batch, so no need to randomize here
        self.num_specs = {}
        index = 0
        for i in range(len(self.classes)):
            self.num_specs[self.classes[i]] = 0
            results = self.db.get_recording_by_subcat_name(self.classes[i])
            for r in results:
                results2 = self.db.get_spectrogram('RecordingID', r.id)
                for r2 in results2:
                    self.x_train[index] = r2.value
                    self.train_class[index] = self.classes[i]
                    self.y_train[index][i] = 1
                    self.num_specs[self.classes[i]] += 1
                    index += 1

        self.create_model()

        # create output directory
        self.out_dir = 'summary'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if cfg.verbosity > 0 and self.model is not None:
            # output text and graphical descriptions of the model
            with open(f'{self.out_dir}/table.txt','w') as text_output:
                self.model.summary(print_fn=lambda x: text_output.write(x + '\n'))

            if cfg.verbosity >= 2:
                keras.utils.plot_model(self.model, show_shapes=True, show_dtype=True, to_file=f'{self.out_dir}/model.png')

        # initialize callbacks
        lr_scheduler = keras.callbacks.LearningRateScheduler(cos_lr_schedule)
        ckpt_path = os.path.join('data', cfg.low_band_ckpt_name)
        self.model_checkpoint_callback = model_checkpoint.ModelCheckpoint(ckpt_path, cfg.ckpt_min_epochs,
            cfg.ckpt_min_val_accuracy, copy_ckpt=cfg.copy_ckpt, save_best_only=cfg.save_best_only)
        self.callbacks = [lr_scheduler, self.model_checkpoint_callback]

        # create the training dataset
        options = tf.data.Options()
        self.datagen = data_generator.DataGenerator(self.db, self.x_train, self.y_train, self.train_class)
        train_ds = tf.data.Dataset.from_generator(
            self.datagen,
            output_types=(tf.float16, tf.float16),
            output_shapes=([cfg.low_band_spec_height, cfg.spec_width, 1],[len(self.classes)]))
        train_ds = train_ds.with_options(options)
        self.train_ds = train_ds.batch(cfg.batch_size)

# learning rate schedule with cosine decay
def cos_lr_schedule(epoch):
    base_lr = cfg.base_lr * cfg.batch_size / 32
    lr = base_lr * (1 + math.cos(epoch * math.pi / max(cfg.num_epochs + cfg.cos_decay_pad, 1))) / 2

    return lr

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=22, help=f'Minimum epochs before saving checkpoint.')
    parser.add_argument('-e', type=int, default=22, help=f'Number of epochs.')
    parser.add_argument('-f', type=str, default=cfg.training_db, help=f'Name of training database. Default = {cfg.training_db}.')
    parser.add_argument('-v', type=int, default=1, help='Verbosity (0-2, 0 is minimal, 2 includes graph of model). Default = 1.')

    args = parser.parse_args()

    cfg.ckpt_min_epochs = args.c
    cfg.num_epochs = args.e
    cfg.training_db = args.f
    cfg.verbosity = args.v

    cfg.augmentation = False
    cfg.spec_height = cfg.low_band_spec_height
    #cfg.prob_real_noise = 0

    if cfg.verbosity > 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', datefmt='%H:%M:%S')

    #keras.mixed_precision.set_global_policy("mixed_float16")

    trainer = Trainer()
    trainer.run()
