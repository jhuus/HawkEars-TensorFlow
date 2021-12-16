# After specified number of epochs save the model whenever val_accuracy is greater 
# than specified amount and also greater than last saved value.

import shutil

import tensorflow as tf
from tensorflow import keras

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, path, min_epochs, min_val_accuracy, copy_ckpt):
        self.path = path
        self.min_epochs = min_epochs
        self.min_val_accuracy = min_val_accuracy
        self.copy_ckpt = copy_ckpt
        self.saved_val_accuracy = 0
        self.best_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs['val_accuracy']
        self.best_val_accuracy = max(val_accuracy, self.best_val_accuracy)
        
        if epoch >= self.min_epochs and val_accuracy >= self.min_val_accuracy \
        and val_accuracy - self.saved_val_accuracy > .0001:
            print(f'\nSaving model checkpoint with val_accuracy = {val_accuracy:.4f}')
            self.model.save(self.path)
            self.saved_val_accuracy = val_accuracy
            
            # this lets us compare results at different epochs on additional data later
            if self.copy_ckpt:
                shutil.copytree(self.path, f'{self.path}-e{epoch + 1}-{val_accuracy:.4f}')
        