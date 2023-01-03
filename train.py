# Train the selected neural network model on spectrograms for birds and a few other classes.
# To see command-line arguments, run the script with -h argument.

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import time
from collections import namedtuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = no info, 2 = no warnings, 3 = no errors
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow import keras

from core import config as cfg
from core import data_generator
from core import database
from core import plot
from core import util

from model import model_checkpoint
from model import efficientnet_v2

class Trainer:
    def __init__(self):
        self.db = database.Database(f'data/{cfg.training_db}.db')
        self.classes = util.get_class_list(cfg.classes_file)
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

    def run(self):
        #strategy = tf.distribute.MirroredStrategy() # for multi-GPU training
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            if cfg.load_saved_model:
                model = keras.models.load_model(cfg.ckpt_path)
            else:
                class_act = 'sigmoid' if cfg.multi_label else 'softmax'
                spec_height = cfg.lnd_spec_height if cfg.low_noise_detector else cfg.spec_height
                model = efficientnet_v2.EfficientNetV2(
                        model_type=cfg.eff_config,
                        num_classes=len(self.classes),
                        input_shape=(spec_height, cfg.spec_width, 1),
                        activation='swish',
                        classifier_activation=class_act,
                        dropout=0.15,
                        drop_connect_rate=0.25)

            opt = keras.optimizers.Adam(learning_rate = cos_lr_schedule(0))
            if cfg.multi_label:
                loss = keras.losses.BinaryCrossentropy(label_smoothing = cfg.label_smoothing)
                model.compile(loss=loss, optimizer=opt)
            else:
                loss = keras.losses.CategoricalCrossentropy(label_smoothing = cfg.label_smoothing)
                model.compile(loss=loss, optimizer=opt, metrics='accuracy')

        # create output directory
        dir = 'summary'
        if not os.path.exists(dir):
            os.makedirs(dir)

        if cfg.verbosity > 0:
            # output text and graphical descriptions of the model
            with open(f'{dir}/table.txt','w') as text_output:
                model.summary(print_fn=lambda x: text_output.write(x + '\n'))

            if cfg.verbosity >= 2:
                keras.utils.plot_model(model, show_shapes=True, to_file=f'{dir}/graphic.png')

        # initialize callbacks
        lr_scheduler = keras.callbacks.LearningRateScheduler(cos_lr_schedule)
        model_checkpoint_callback = model_checkpoint.ModelCheckpoint(cfg.ckpt_path, cfg.ckpt_min_epochs,
            cfg.ckpt_min_val_accuracy, copy_ckpt=cfg.copy_ckpt, save_best_only=cfg.save_best_only)
        callbacks = [lr_scheduler, model_checkpoint_callback]

        # create the training and test datasets
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        datagen = data_generator.DataGenerator(self.db, self.x_train, self.y_train, self.train_class)
        train_ds = tf.data.Dataset.from_generator(
            datagen,
            output_types=(tf.float16, tf.float16),
            output_shapes=([self.spec_height, cfg.spec_width, 1],[len(self.classes)]))
        train_ds = train_ds.with_options(options)
        train_ds = train_ds.batch(cfg.batch_size)

        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        test_ds = test_ds.with_options(options)
        test_ds = test_ds.batch(cfg.batch_size)

        # run training
        start_time = time.time()
        history = model.fit(train_ds, epochs = cfg.num_epochs, verbose = cfg.verbosity, validation_data = test_ds,
            shuffle = False, callbacks = callbacks, class_weight = self._get_class_weight())
        elapsed = time.time() - start_time

        if cfg.verbosity > 0:
            with open(f'{dir}/summary.txt','w') as text_output:
                text_output.write(f'EfficientNetV2 config: {cfg.eff_config}\n')
                text_output.write(f'Batch size: {cfg.batch_size}\n')
                text_output.write(f'Epochs: {cfg.num_epochs}\n')

                if 'loss' in history.history:
                    text_output.write(f"Training loss: {history.history['loss'][-1]:.3f}\n")

                if not cfg.multi_label and 'accuracy' in history.history:
                    # output accuracy graph
                    training_accuracy = history.history['accuracy'][-1]
                    text_output.write(f'Training accuracy: {training_accuracy:.3f}\n')

                    if len(self.x_test) > 0:
                        self.plot_results(dir, history, 'accuracy', 'val_accuracy')
                        scores = model.evaluate(self.x_test, self.y_test)
                        test_accuracy = scores[1]
                    else:
                        self.plot_results(dir, history, 'accuracy')


                if cfg.verbosity >= 1 and len(self.x_test) > 0:
                    text_output.write(f'Test loss: {scores[0]:.3f}\n')
                    text_output.write(f'Final test accuracy: {test_accuracy:.3f}\n')
                    text_output.write(f'Best test accuracy: {model_checkpoint_callback.best_val_accuracy:.4f}\n')

                minutes = int(elapsed) // 60
                seconds = int(elapsed) % 60
                text_output.write(f'Elapsed time for training = {minutes}m {seconds}s\n')

            if cfg.verbosity >= 1 and 'loss' in history.history:
                # output loss graph
                if len(self.x_test) > 0:
                    self.plot_results(dir, history, 'loss', 'val_loss')
                else:
                    self.plot_results(dir, history, 'loss')

            if len(self.x_test) > 0:
                print(f'Best test accuracy: {model_checkpoint_callback.best_val_accuracy:.4f}\n')

            print(f'Elapsed time for training = {minutes}m {seconds}s\n')

        return model_checkpoint_callback.best_val_accuracy

    # given the total number of spectrograms in a class, return a dict of randomly selected
    # indices to use for testing (indices not in the list are used for training)
    def get_test_indices(self, total):
        num_test = math.ceil(cfg.test_portion * total)

        test_indices = {}
        while len(test_indices.keys()) < num_test:
            index = random.randint(0, total - 1)
            if index not in test_indices:
                test_indices[index] = 1

        return test_indices

    # calculate and return class weights
    def _get_class_weight(self):
        num_specs = {}
        sum = 0
        for class_name in self.classes:
            count = float(self.db.get_spectrogram_count(class_name))
            if count == 0:
                print(f'Error. No spectrograms found for "{class_name}"')
                quit()

            num_specs[class_name] = count
            sum += count

        class_weight = {}
        average = sum / len(self.classes)
        avg_sqrt = math.sqrt(average)
        for i, class_name in enumerate(self.classes):
            weight = avg_sqrt / math.sqrt(num_specs[class_name])

            print(f'Applying weight {weight:.2f} to {class_name}')
            class_weight[i] = weight

        return class_weight

    def init(self):
        if cfg.low_noise_detector:
            self.spec_height = cfg.lnd_spec_height
        else:
            self.spec_height = cfg.spec_height

        # count spectrograms and randomly select which to use for testing vs. training
        num_spectrograms = []
        self.test_indices = []
        for i in range(len(self.classes)):
            total = self.db.get_spectrogram_count(self.classes[i])
            num_spectrograms.append(total)
            self.test_indices.append(self.get_test_indices(total))

        # get the total training and testing counts across all classes
        test_total = 0
        train_total = 0
        for i in range(len(self.classes)):
            test_count = len(self.test_indices[i].keys())
            train_count = num_spectrograms[i] - test_count
            test_total += test_count
            train_total += train_count

        print(f'# training samples: {train_total}, # test samples: {test_total}')

        # initialize arrays
        self.x_train = [0 for i in range(train_total)]
        self.y_train = np.zeros((train_total, len(self.classes)))
        self.train_class = ['' for i in range(train_total)]
        self.x_test = np.zeros((test_total, self.spec_height, cfg.spec_width, 1))
        self.y_test = np.zeros((test_total, len(self.classes)))
        self.input_shape = (self.spec_height, cfg.spec_width, 1)

        # map test spectrogram indexes to file names for outputting names of misidentified ones
        self.spec_file_name = {}

        # populate from the database;
        # they will be selected randomly per mini batch, so no need to randomize here
        train_index = 0
        test_index = 0

        for i in range(len(self.classes)):
            results = self.db.get_recording_by_subcat_name(self.classes[i])
            spec_index = 0
            for r in results:
                results2 = self.db.get_spectrogram('RecordingID', r.id)
                for r2 in results2:
                    if spec_index in self.test_indices[i]:
                        # test spectrograms are expanded here
                        self.spec_file_name[test_index] = f'{r.filename}-{r2.offset}' # will be used in names of files written to misident folder
                        self.x_test[test_index] = util.expand_spectrogram(r2.value, low_noise_detector=cfg.low_noise_detector)
                        self.y_test[test_index][i] = 1
                        test_index += 1
                    else:
                        # training spectrograms are expanded in data generator
                        self.x_train[train_index] = r2.value
                        self.train_class[train_index] = self.classes[i]
                        self.y_train[train_index][i] = 1
                        train_index += 1

                    spec_index += 1

# learning rate schedule with cosine decay
def cos_lr_schedule(epoch):
    base_lr = cfg.base_lr * cfg.batch_size / 64
    lr = base_lr * (1 + math.cos(epoch * math.pi / max(cfg.num_epochs, 1))) / 2

    if cfg.verbosity == 0:
        print(f'epoch: {epoch + 1} / {cfg.num_epochs}') # so there is at least some status info

    return lr

if __name__ == '__main__':

    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=cfg.ckpt_min_epochs, help=f'Minimum epochs before saving checkpoint. Default = {cfg.ckpt_min_epochs}.')
    parser.add_argument('-e', type=int, default=cfg.num_epochs, help=f'Number of epochs. Default = {cfg.num_epochs}.')
    parser.add_argument('-f', type=str, default=cfg.training_db, help=f'Name of training database. Default = {cfg.training_db}.')
    parser.add_argument('-m', type=int, default=0, help='If 1, load a saved model, else train from scratch. Default = 0.')
    parser.add_argument('-g', type=str, default=cfg.eff_config, help=f'Name of EfficientNet_V2 configuration to use. Default = {cfg.eff_config}.')
    parser.add_argument('-r', type=float, default=cfg.base_lr, help=f'Base learning rate. Default = {cfg.base_lr}.')
    parser.add_argument('-t', type=float, default=cfg.test_portion, help=f'Test portion. Default = {cfg.test_portion}')
    parser.add_argument('-u', type=int, default=1, help='1 = If 1, train a multi-label classifier. Default = 1.')
    parser.add_argument('-v', type=int, default=1, help='Verbosity (0-2, 0 is minimal, 2 includes graph of model). Default = 1.')
    parser.add_argument('-y', type=int, default=0, help='If y = 1, extract spectrograms for low-noise detector. Default = 0.')
    parser.add_argument('-z', type=int, default=cfg.seed, help=f'Integer seed for random number generators. Default = {cfg.seed}. If specified, other settings to increase repeatability will also be enabled, which slows down training.')

    args = parser.parse_args()

    cfg.ckpt_min_epochs = args.c
    cfg.num_epochs = args.e
    cfg.training_db = args.f
    cfg.eff_config = args.g
    cfg.load_saved_model = (args.m == 1)
    cfg.base_lr = args.r
    cfg.test_portion = args.t
    cfg.multi_label = (args.u == 1)
    cfg.verbosity = args.v
    cfg.low_noise_detector = (args.y == 1)
    cfg.seed = args.z

    if cfg.seed != None:
        # these settings make results more reproducible, which is very useful when tuning parameters
        os.environ['PYTHONHASHSEED'] = str(cfg.seed)
        # tf.config.experimental.enable_op_determinism() # uncomment this for more determinism, but MUCH slower
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    # mixed precision trains 25-30% faster, but limits portability
    if cfg.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    trainer = Trainer()
    trainer.run()
