# Train the selected neural network model on spectrograms for birds and a few other classes.
# Train the selected neural network model on spectrograms for birds and a few other classes.
# To see command-line arguments, run the script with -h argument.

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import sys
import time
import zlib
from collections import namedtuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 = no info, 2 = no warnings, 3 = no errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras

from core import audio
from core import constants
from core import data_generator
from core import database
from core import util

from model import model_checkpoint
from model import resnest

class Trainer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.db = database.Database(f'data/{parameters.training}.db')
        self.classes = util.get_class_list()
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
        # only use MirroredStrategy in a multi-GPU environment
        #strategy = tf.distribute.MirroredStrategy()
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            # define and compile the model
            if self.parameters.type == 0:
                model = keras.models.load_model(constants.CKPT_PATH)
            else:
                model_builder = resnest.ResNest(num_classes=len(self.classes), 
                        input_shape=(self.spec_height, constants.SPEC_WIDTH, 1),
                        num_stages=self.parameters.num_stages,
                        blocks_set=self.parameters.blocks_per_stage[:self.parameters.num_stages],
                        kernel_size=self.parameters.kernel_size,
                        seed=self.parameters.seed)
                model = model_builder.build_model()

            opt = keras.optimizers.Adam(learning_rate = cos_lr_schedule(0))
            loss = keras.losses.CategoricalCrossentropy(label_smoothing = 0.0)
            model.compile(loss = loss, optimizer = opt, metrics = 'accuracy') 

        # create output directory
        dir = 'summary'
        if not os.path.exists(dir):
            os.makedirs(dir)
          
        # output text and graphical descriptions of the model
        with open(f'{dir}/table.txt','w') as text_output:
            model.summary(print_fn=lambda x: text_output.write(x + '\n'))
        
        if self.parameters.verbosity == 3:
            keras.utils.plot_model(model, show_shapes=True, to_file=f'{dir}/graphic.png')

        # initialize callbacks
        lr_scheduler = keras.callbacks.LearningRateScheduler(cos_lr_schedule) 
        model_checkpoint_callback = model_checkpoint.ModelCheckpoint(
            self.parameters.ckpt_path, self.parameters.ckpt_min_epochs, self.parameters.ckpt_min_val_accuracy, self.parameters.copy_ckpt)
        callbacks = [lr_scheduler, model_checkpoint_callback] 
          
        # create the training and test datasets
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        datagen = data_generator.DataGenerator(self.x_train, self.y_train, seed=self.parameters.seed, binary_classifier=self.parameters.binary_classifier)
        train_ds = tf.data.Dataset.from_generator(
            datagen, 
            output_types=(tf.float16, tf.float16), 
            output_shapes=([self.spec_height, constants.SPEC_WIDTH, 1],[len(self.classes)]))
        train_ds = train_ds.with_options(options)
        train_ds = train_ds.batch(self.parameters.batch_size)
        
        test_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        test_ds = test_ds.with_options(options)
        test_ds = test_ds.batch(self.parameters.batch_size)
        
        class_weight = self._get_class_weight()
        
        # run training
        if self.parameters.seed is None:
            workers = 2
        else:
            workers = 0 # run data augmentation in main thread to improve repeatability
        
        start_time = time.time()
        history = model.fit(train_ds, epochs = self.parameters.epochs, verbose = self.parameters.verbosity, validation_data = test_ds,
            workers = workers, shuffle = False, callbacks = callbacks, class_weight = class_weight)
        elapsed = time.time() - start_time
          
        # output loss/accuracy graphs and a summary report
        scores = model.evaluate(self.x_test, self.y_test) 

        self.plot_results(dir, history, 'accuracy')
        self.plot_results(dir, history, 'loss')
        
        training_accuracy = history.history["accuracy"][-1]
        test_accuracy = scores[1]
        
        # report on misidentified test spectrograms
        predictions = model.predict(self.x_test)
        self.analyze_predictions(predictions)
        
        with open(f'{dir}/summary.txt','w') as text_output:
            text_output.write(f'Number of stages: {self.parameters.num_stages}\n')
            text_output.write(f'Blocks per stage: {self.parameters.blocks_per_stage}\n')
            text_output.write(f'Batch size: {self.parameters.batch_size}\n')
            text_output.write(f'Epochs: {self.parameters.epochs}\n')
            text_output.write(f'Training loss: {history.history["loss"][-1]:.3f}\n')
            text_output.write(f'Training accuracy: {training_accuracy:.3f}\n')
            text_output.write(f'Test loss: {scores[0]:.3f}\n')
            text_output.write(f'Final test accuracy: {test_accuracy:.3f}\n')
            text_output.write(f'Best test accuracy: {model_checkpoint_callback.best_val_accuracy:.4f}\n')
            
            minutes = int(elapsed) // 60
            seconds = int(elapsed) % 60
            text_output.write(f'Elapsed time for training = {minutes}m {seconds}s\n')
            
            print(f'Best test accuracy: {model_checkpoint_callback.best_val_accuracy:.4f}\n')
            print(f'Elapsed time for training = {minutes}m {seconds}s\n')
            
        return training_accuracy, test_accuracy, elapsed
        
    # find and report on incorrect predictions;
    # always generate summary/stats.csv, but output misident/*.png only if verbosity >= 2;
    # this is based on the last epoch, which may not be the best saved model
    def analyze_predictions(self, predictions):
        class ClassInfo:
            def __init__(self):
                self.spec_count = 0
                self.true_pos = 0
                self.false_pos = 0
                self.false_neg = 0
    
        if self.parameters.verbosity >= 2:
            misident_dir = 'misident'
            if os.path.exists(misident_dir):
                shutil.rmtree(misident_dir) # ensure we start with an empty folder
                
            os.makedirs(misident_dir)
    
        # collect data per class and output images if requested
        classes = {}
        for i in range(len(predictions)):
            actual_index = np.argmax(self.y_test[i])
            actual_name = self.classes[actual_index]

            predicted_index = np.argmax(predictions[i])
            predicted_name = self.classes[predicted_index]
            
            if actual_name in classes:
                actual_class_info = classes[actual_name]
            else:
                actual_class_info = ClassInfo()
                
                classes[actual_name] = actual_class_info
            
            if predicted_name in classes:
                predicted_class_info = classes[predicted_name]
            else:
                predicted_class_info = ClassInfo()
                classes[predicted_name] = predicted_class_info
            
            actual_class_info.spec_count += 1
            if predicted_index == actual_index:
                actual_class_info.true_pos += 1
            else:
                actual_class_info.false_neg += 1
                predicted_class_info.false_pos += 1
                
                if self.parameters.verbosity >= 2:
                    if i in self.spec_file_name.keys():  
                        suffix = self.spec_file_name[i]
                    else:
                        suffix = i
                    
                    spec = self.x_test[i].reshape(self.x_test[i].shape[0], self.x_test[i].shape[1])
                    util.plot_spec(spec, f'{misident_dir}/{actual_name}_{predicted_name}_{suffix}.png')
                
        # output stats.csv containing data per class
        stats = 'class,count,TP,FP,FN,FP+FN,precision,recall,average\n'
        for class_name in sorted(classes):
            count = classes[class_name].spec_count
            tp = classes[class_name].true_pos
            fp = classes[class_name].false_pos
            fn = classes[class_name].false_neg
            
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
                
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            
            stats += f'{class_name},{count},{tp:.3f},{fp:.3f},{fn:.3f},{fp + fn:.3f},{precision:.3f},{recall:.3f},{(precision+recall)/2:.3f}\n'
        
        with open(f'summary/stats.csv','w') as text_output:
            text_output.write(stats)

    # given the total number of spectrograms in a class, return a dict of randomly selected
    # indices to use for testing (indices not in the list are used for training)
    def get_test_indices(self, total):
        num_test = math.ceil(self.parameters.test_portion * total)
        
        test_indices = {}
        while len(test_indices.keys()) < num_test:
            index = random.randint(0, total - 1)
            if index not in test_indices.keys():
                test_indices[index] = 1
        
        return test_indices
        
    # heuristic to adjust weights of classes;
    # data/weights.txt contains optional weight per class name;
    # format is "class-name,weight", e.g. "Noise,1.1";
    # classes not listed there default to a weight of 1.0
    def _get_class_weight(self):
        input_weight = {}
        path = 'data/weights.txt'
        try:
            with open(path, 'r') as file:
                for line in file.readlines():
                    line = line.strip()
                    if len(line) > 0 and line[0] != '#':
                        tokens = line.split(',')
                        if len(tokens) > 1:
                            try:
                                weight = float(tokens[1])
                                input_weight[tokens[0].strip()] = weight
                            except ValueError:
                                print(f'Invalid input weight = {tokens[1]} for class {tokens[0]}')
        except IOError:
            print(f'Unable to open weights file "{path}"')
            return
    
        class_weight = {}
        for i in range(len(self.classes)):
            if self.classes[i] in input_weight.keys():
                print(f'Assigning weight {input_weight[self.classes[i]]} to {self.classes[i]}')
                class_weight[i] = input_weight[self.classes[i]]
            else:
                class_weight[i] = 1.0
        
        return class_weight

    def init(self):
        if self.parameters.binary_classifier:
            self.spec_height = constants.BINARY_SPEC_HEIGHT
        else:
            self.spec_height = constants.SPEC_HEIGHT
    
        # count spectrograms and randomly select which to use for testing vs. training
        num_spectrograms = []
        self.test_indices = []
        for i in range(len(self.classes)):
            total = self.db.get_num_spectrograms(self.classes[i])
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
            
        if len(self.parameters.val_db) > 0:
            # when we just use a portion of the training data for testing/validation, it ends up being highly
            # correlated with the training data, so the validation percentage is artificially high and it's
            # difficult to detect overfitting;
            # adding separate test data from a validation database helps to counteract this;
            # there can be multiple, which must be comma-separated
            val_names = self.parameters.val_db.split(',')
            for val_name in val_names:
                validation_db = database.Database(f'data/{val_name}.db')
                num_validation_specs = 0
                for class_name in self.classes:
                    test_total += validation_db.get_num_spectrograms(class_name)
            
        print(f'# training samples: {train_total}, # test samples: {test_total}')

        # initialize arrays
        self.x_train = [0 for i in range(train_total)]
        self.y_train = np.zeros((train_total, len(self.classes)))
        self.x_test = np.zeros((test_total, self.spec_height, constants.SPEC_WIDTH, 1))
        self.y_test = np.zeros((test_total, len(self.classes)))
        self.input_shape = (self.spec_height, constants.SPEC_WIDTH, 1)
        
        # map test spectrogram indexes to file names for outputting names of misidentified ones
        self.spec_file_name = {}
        
        # populate from the database;
        # they will be selected randomly per mini batch, so no need to randomize here 
        train_index = 0
        test_index = 0
        
        for i in range(len(self.classes)):
            results = self.db.get_recordings_by_subcategory_name(self.classes[i])
            spec_index = 0
            for result in results:
                recording_id, file_name = result
                specs = self.db.get_spectrograms_by_recording_id(recording_id)
               
                for j in range(len(specs)):
                    spec, offset = specs[j]
                    if spec_index in self.test_indices[i].keys():
                        # test spectrograms are expanded here
                        self.spec_file_name[test_index] = f'{file_name}-{offset}' # will be used in names of files written to misident folder
                        self.x_test[test_index] = util.expand_spectrogram(spec, binary_classifier=self.parameters.binary_classifier)
                        self.y_test[test_index][i] = 1
                        test_index += 1
                    else:
                        # training spectrograms are expanded in data generator
                        self.x_train[train_index] = spec
                        self.y_train[train_index][i] = 1
                        train_index += 1
                        
                    spec_index += 1
                    
        if len(self.parameters.val_db) > 0:
            # append test data from the validation database(s)
            val_names = self.parameters.val_db.split(',')
            for val_name in val_names:
                validation_db = database.Database(f'data/{val_name}.db')
                for i in range(len(self.classes)):
                    specs = validation_db.get_spectrograms_by_name(self.classes[i])
                    for spec in specs:
                        self.x_test[test_index] = util.expand_spectrogram(spec[0], binary_classifier=self.parameters.binary_classifier)
                        self.y_test[test_index][i] = 1
                        test_index += 1

# learning rate schedule with cosine decay
def cos_lr_schedule(epoch):
    global trainer
    base_lr = trainer.parameters.base_lr * trainer.parameters.batch_size / 64
    lr = base_lr * (1 + math.cos(epoch * math.pi / max(trainer.parameters.epochs, 1))) / 2
    
    if trainer.parameters.verbosity == 0:
        print(f'epoch: {epoch + 1} / {trainer.parameters.epochs}') # so there is at least some status info
        
    return lr

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, help='Model type (0 = Load existing model, 1 = ResNeSt. Default = 1.')
    parser.add_argument('-s', type=int, default=1, choices=[1, 2, 3, 4], help='Number of stages (a stage is a group of layers that use the same feature map size). Default = 1, max = 4.')
    parser.add_argument('-n1', type=int, default=1, help='Number of blocks in first stage. Default = 1.')
    parser.add_argument('-n2', type=int, default=1, help='Number of blocks in second stage. Default = 1.')
    parser.add_argument('-n3', type=int, default=1, help='Number of blocks in third stage. Default = 1.')
    parser.add_argument('-n4', type=int, default=1, help='Number of blocks in fourth stage. Default = 1.')
    parser.add_argument('-b', type=int, default=32, help='Batch size. Default = 32.')
    parser.add_argument('-c', type=int, default=0, help='Minimum epochs before saving checkpoint. Default = 0.')
    parser.add_argument('-d', type=float, default=0.9, help='Minimum validation accuracy before saving checkpoint. Default = 0.90.')
    parser.add_argument('-e', type=int, default=10, help='Number of epochs. Default = 10.')
    parser.add_argument('-f', type=str, default='training', help='Name of training database. Default = training.')
    parser.add_argument('-k', type=int, default=3, help='Resnest kernel size. Default = 3.')
    parser.add_argument('-r', type=float, default=.006, help='Base learning rate. Default = .006')
    parser.add_argument('-t', type=float, default=.01, help='Test portion. Default = .01')
    parser.add_argument('-v', type=int, default=1, help='Verbosity (0-2, 0 omits output graphs, 2 plots misidentified test spectrograms, 3 adds graph of model). Default = 1.')
    parser.add_argument('-x', type=str, default='', help='Name(s) of extra validation databases. "abc" means load "abc.db". "abc,def" means load both databases for validation. Default = "". ')
    parser.add_argument('-y', type=int, default=0, help='If y = 1, extract spectrograms for binary classifier. Default = 0.')
    parser.add_argument('-z', type=int, default=None, help='Integer seed for random number generators. Default = None (do not). If specified, other settings to increase repeatability will also be enabled, which slow down training.')
    args = parser.parse_args()
    
    Parameters = namedtuple('Parameters', ['type', 'num_stages', 'blocks_per_stage', 'epochs', 'base_lr', 'batch_size', 'kernel_size', 
                            'test_portion', 'val_db', 'verbosity', 'ckpt_path', 'ckpt_min_epochs', 'ckpt_min_val_accuracy', 'copy_ckpt', 
                            'seed', 'training', 'binary_classifier'])
    parameters = Parameters(type = args.m, num_stages = args.s, blocks_per_stage = [args.n1, args.n2, args.n3, args.n4], epochs = args.e, base_lr=args.r, batch_size = args.b, 
                            kernel_size=args.k, test_portion = args.t, val_db = args.x, verbosity = args.v, ckpt_path=constants.CKPT_PATH, ckpt_min_epochs=args.c, 
                            ckpt_min_val_accuracy=args.d, copy_ckpt=True, seed=args.z, training=args.f, binary_classifier=(args.y==1))
                            
    if args.z != None:
        # these settings make results more reproducible, which is very useful when tuning parameters
        os.environ['PYTHONHASHSEED'] = str(args.z)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(args.z)
        np.random.seed(args.z)
        tf.random.set_seed(args.z)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    global trainer
    trainer = Trainer(parameters)
    trainer.run()