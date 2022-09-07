# audio
segment_len = 3         # duration per spectrogram in seconds
sampling_rate = 44100
n_fft = 2048
min_freq = 30
max_freq = 12500
low_noise_spec_height = 32 # spectrogram height used by low noise detector
spec_height = 128       # normal spectrogram height
spec_width = 384        # spectrogram width

# training
base_lr = .006          # base learning rate
batch_size = 32
eff_config = 'a3'       # EfficientNet_v2 configuration to use
ckpt_min_epochs = 15    # min epochs before saving checkpoint
ckpt_min_val_accuracy = 0 # min validation accuracy before saving checkpoint
copy_ckpt = True        # save a copy of each checkpoint
load_saved_model = False
low_noise_detector = False # train the low-noise detector
mixed_precision = False
multi_label = True
num_epochs = 18
save_best_only = False
seed = None
test_portion = .01
training_db = 'training' # name of training database
validation_db = ''       # name of optional validation database
verbosity = 1            # 0 omits output graphs, 2 adds plots of misidentified spectrograms, 3 adds graph of model
apply_sqrt_to_weights=True # reduce class weight differences, e.g. [.64, 1] becomes [.8, 1]
min_class_weight = 0.4
max_class_weight = 2.5

# miscellaneous
reset_model_counter = 10            # in analysis, reset the model every n loops to avoid running out of GPU memory
ignore_file = 'data/ignore.txt'     # classes listed in this file are ignored in analysis
classes_file = 'data/classes.txt'   # list of classes used in training and analysis
ckpt_path = 'data/ckpt'             # main model checkpoint
denoiser_path = 'data/denoiser'     # model checkpoint for denoiser
low_noise_ckpt_path = 'data/is_low_noise_ckpt' # model checkpoint for low noise detector
