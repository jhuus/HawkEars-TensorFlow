# Save the model when specified by the parameters.

import logging
import os
import shutil

import tensorflow as tf
from tensorflow import keras

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, path, min_epochs, min_val_accuracy, copy_ckpt=False, save_best_only=True):
        self.path = path
        self.min_epochs = min_epochs
        self.min_val_accuracy = min_val_accuracy
        self.copy_ckpt = copy_ckpt
        self.saved_val_accuracy = 0
        self.best_val_accuracy = 0
        self.save_best_only = save_best_only

    def on_epoch_end(self, epoch, logs=None):
        if 'val_accuracy' in logs.keys():
            val_accuracy = logs['val_accuracy']
        else:
            val_accuracy = 0

        self.best_val_accuracy = max(val_accuracy, self.best_val_accuracy)

        if epoch >= self.min_epochs - 1 and val_accuracy >= self.min_val_accuracy \
        and (not self.save_best_only or (val_accuracy - self.saved_val_accuracy > .0001)):
            print()
            logging.info(f'Saving model checkpoint with val_accuracy = {val_accuracy:.4f}')
            self.model.save(self.path)
            self.saved_val_accuracy = val_accuracy

            # this lets us compare results at different epochs on additional data later
            if self.copy_ckpt:
                dest_path = f'{self.path}-e{epoch + 1}'
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)

                shutil.copytree(self.path, dest_path)
