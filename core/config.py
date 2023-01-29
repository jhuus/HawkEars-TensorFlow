# basic audio
segment_len = 3         # spectrogram duration in seconds
sampling_rate = 40960
hop_length = 320        # FFT parameter
win_length = 2048       # FFT parameter
spec_height = 128       # spectrogram height
spec_width = 384        # spectrogram width (3 * 128)
min_audio_freq = 200
max_audio_freq = 10500
mel_scale = True
mel_amplitude_adjustment = True # correct for amplitude distortion caused by mel scaling
spec_exponent = .68      # raise spectrogram values to this exponent (brings out faint sounds)
spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

# training
seed = 1
base_lr = .003          # base learning rate
cos_decay_pad = 0       # larger values reduce cosine decay of learning rate
batch_size = 32
eff_config = 'a3'       # EfficientNet_v2 configuration to use ('r' for ResNest)
eff_dropout = 0.4       # dropout rate for EfficientNext_V2 output block
eff_drop_connect = 0.5  # dropout rate for EfficientNext_V2 non-output blocks
resnest_stages = 2      # if eff_config = 'r'
resnest_blocks = [2,2]
num_epochs = 1
ckpt_min_epochs =  5    # min epochs before saving checkpoint
ckpt_min_val_accuracy = 0 # min validation accuracy before saving checkpoint
copy_ckpt = True        # save a copy of each checkpoint
label_smoothing = 0.15

load_saved_model = False
mixed_precision = True  # mixed precision trains faster with large models, but slower with tiny models
multi_label = True
save_best_only = False
deterministic = False    # may reduce variance a bit, but still not deterministic
test_portion = 0
frequency_db = 'frequency' # eBird barchart data, i.e. species report frequencies
training_db = 'training' # name of training database
verbosity = 1            # 0 omits output graphs, 2 adds plots of misidentified spectrograms, 3 adds graph of model

# data augmentation
augmentation = True
prob_merge = 0.35        # probability of merging to train multi-label support
prob_aug = 0.5           # probability of augmenting after merge and before fade
white_noise_weight = 1.0 # weights give relative probability of each augmentation type
shift_weight = 0.02
speckle_weight = 0.04
max_shift = 5            # max pixels for horizontal shift
white_noise_variance = 0.001 # larger variances lead to more noise
speckle_variance = .009
min_fade = 0.1           # multiply values by a random float in [min_fade, max_fade]
max_fade = 1.0

# analysis / inference
min_prob = 0.75              # minimum confidence level
use_banding_codes = True     # use banding codes instead of species names in labels
check_adjacent = True        # omit label unless adjacent segment matches
adjacent_prob_factor = 0.65  # when checking if adjacent segment matches species, use self.min_prob times this
reset_model_counter = 10     # in analysis, reset the model every n loops to avoid running out of GPU memory
top_n = 6 # number of top matches to log in debug mode
min_location_freq = .0001    # ignore if species frequency less than this for location/week
file_date_regex = '\S+_(\d+)_.*' # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
file_date_regex_group = 1    # use group at offset 1

# Soundalike groups are used in analysis / inference when a location is given.
# For each soundalike species, eBird barchart data is accessed to get the maximum
# frequency across all weeks (i.e. maximum portion of checklists that include the species).
# If the maximum frequency for a species in a soundalike group is <= soundalike_cutoff,
# it is replaced by the species with the highest frequency > soundalike_cutoff in the group.
# For instance, if a Mountain Chickadee is ID'd in a county where it's never been seen,
# but Black-capped Chickadees are common there, it will be reported as a Black-capped Chickadee.
soundalike_cutoff = .005
soundalikes = [['Black-capped Chickadee', 'Boreal Chickadee', 'Mountain Chickadee'],
               ['Pacific Wren', 'Winter Wren']]

# paths
ignore_file = 'data/ignore.txt'     # classes listed in this file are ignored in analysis
classes_file = 'data/classes.txt'   # list of classes used in training and analysis
ckpt_path = 'data/ckpt'             # main model checkpoint
