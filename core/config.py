# basic audio
segment_len = 3         # spectrogram duration in seconds
sampling_rate = 40960   # 20 * 2048; with win_len and hop_len this ensures correct spectrogram widths
win_length = 2048
hop_length = 320
spec_height = 128       # normal spectrogram height
spec_width = 384        # spectrogram width
min_freq = 30
max_freq = 12500
mel_scale = True        # use False (linear scale) only for plotting spectrograms
spec_exponent=.80       # raise spectrogram values to this exponent
spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

# low noise detector
lnd_spec_height = 32
lnd_chk_idx=20
lnd_max_factor = 1.3
lnd_min_confidence = 0.60

# training
base_lr = .006          # base learning rate
batch_size = 32
eff_config = 'a3'       # EfficientNet_v2 configuration to use
num_epochs = 18
ckpt_min_epochs = 15    # min epochs before saving checkpoint
ckpt_min_val_accuracy = 0 # min validation accuracy before saving checkpoint
copy_ckpt = True        # save a copy of each checkpoint
label_smoothing = 0.13
load_saved_model = False
low_noise_detector = False # True = train the low-noise detector
mixed_precision = True  # trains much faster if true
multi_label = True
save_best_only = False
seed = None
test_portion = 0
frequency_db = 'frequency' # eBird barchart data, i.e. species report frequencies
training_db = 'training' # name of training database
verbosity = 1            # 0 omits output graphs, 2 adds plots of misidentified spectrograms, 3 adds graph of model

# data augmentation
augmentation = True
prob_aug = 0.45         # probability of augmenting a given spectrogram
prob_merge = 0.25       # probability of merging to train multi-label support
max_shift = 5           # max pixels for horizontal shift
noise_variance = 0.0015 # larger variances lead to more noise
speckle_variance = .009
noise_min = 0.1
noise_max = 0.5
min_fade_factor=0.6
max_fade_factor=0.99
min_fade_val=0.05

blur_freq = 0.25        # following are relative frequencies of each augmentation
fade_freq = 0.5
white_noise_freq = 0.1
pink_noise_freq = 0.1
real_noise_freq = 0.8
shift_freq = 0.5
speckle_freq = 0.9

min_mult = 0.8
max_mult = 1.0

# analysis / inference
min_prob = 0.9              # minimum confidence level
use_banding_codes = False   # use banding codes instead of species names in labels
check_adjacent = True       # omit label unless adjacent segment matches
dampen_low_noise = True     # use neural net to identify low frequency noise in audio input, then attenuate it
adjacent_prob_factor = 0.65 # when checking if adjacent segment matches species, use self.min_prob times this
reset_model_counter = 10    # in analysis, reset the model every n loops to avoid running out of GPU memory
top_n = 6 # number of top matches to log in debug mode
min_freq = .01             # ignore if species frequency less than this for location/week
file_date_regex = '\S+_(\d+)_.*' # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
file_date_regex_group = 1   # use group at offset 1

# Soundalike groups are used in analysis / inference when a location is given.
# For each soundalike species, eBird barchart data is accessed to get the maximum
# frequency across all weeks (i.e. maximum portion of checklists that include the species).
# If the maximum frequency for a species in a soundalike group is <= soundalike_cutoff,
# it is replaced by the species with the highest frequency > soundalike_cutoff in the group.
# For instance, if a Pacific Wren is ID'd in a county where it's never been seen,
# but Winter Wrens are common there, it will be reported as a Winter Wren.
soundalike_cutoff = .005
soundalikes = [['Pacific Wren', 'Winter Wren'],
               ['Black-capped Chickadee', 'Boreal Chickadee', 'Mountain Chickadee']]

# paths
ignore_file = 'data/ignore.txt'     # classes listed in this file are ignored in analysis
classes_file = 'data/classes.txt'   # list of classes used in training and analysis
ckpt_path = 'data/ckpt'             # main model checkpoint
bird_ckpt_path = 'data/is_bird_ckpt' # model checkpoint for bird detector
low_noise_ckpt_path = 'data/is_low_noise_ckpt' # model checkpoint for low-frequency noise detector
