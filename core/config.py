# basic audio
segment_len = 3         # spectrogram duration in seconds
sampling_rate = 40960
hop_length = 320        # FFT parameter
win_length = 2048       # FFT parameter
spec_height = 256       # spectrogram height
spec_width = 384        # spectrogram width (3 * 128)
check_seconds = 6       # check prefix of this length when picking cleanest channel
min_audio_freq = 200
max_audio_freq = 10500
mel_scale = True
mel_amplitude_adjustment = True # correct for amplitude distortion caused by mel scaling
spec_exponent = .68      # raise spectrogram values to this exponent (brings out faint sounds)
spec_block_seconds = 240 # max seconds of spectrogram to create at a time (limited by GPU memory)

# low-frequency audio settings for Ruffed Grouse drumming identifier
low_band_spec_height = 64
low_band_min_audio_freq = 0
low_band_max_audio_freq = 200
low_band_mel_scale = False
low_band_ckpt_name = 'ckpt_low_band'

# training
load_saved_model = False
mixed_precision = True  # mixed precision trains faster on GPU but slower on CPU (also may be slower with tiny models on GPU)
multi_label = True
save_best_only = False
deterministic = False    # may reduce variance a bit, but still not deterministic
seed = 1
base_lr = .0025         # base learning rate
cos_decay_pad = 0       # larger values reduce cosine decay of learning rate
batch_size = 32
eff_config = 'a4'       # EfficientNet_v2 configuration to use ('r' for ResNest)
eff_dropout = 0.40      # dropout rate for EfficientNet_V2 output block
eff_drop_connect = 0.50 # dropout rate for EfficientNet_V2 non-output blocks
resnest_stages = 2      # if eff_config = 'r'
resnest_blocks = [2,2]
num_epochs = 2
ckpt_min_epochs =  1    # min epochs before saving checkpoint
ckpt_min_val_accuracy = 0 # min validation accuracy before saving checkpoint
copy_ckpt = True        # save a copy of each checkpoint
label_smoothing = 0.15
test_portion = 0
training_db = 'training' # name of training database
verbosity = 1            # 0 omits output graphs, 2 adds plots of misidentified spectrograms, 3 adds graph of model

# data augmentation
augmentation = True
prob_merge = 0.35        # probability of merging to train multi-label support
prob_aug = 0.5           # probability of augmenting after merge and before fade
prob_speckle = 0.05      # see data_generator.py for usage
prob_real_noise = 0.06
min_white_noise_variance = 0.0009 # larger variances lead to more noise
max_white_noise_variance = 0.0011
speckle_variance = .009
real_noise_factor = 0.2
min_fade = 0.1          # multiply values by a random float in [min_fade, max_fade]
max_fade = 1.0

# analysis / inference
min_prob = 0.80              # minimum confidence level
use_banding_codes = True     # use banding codes instead of species names in labels
check_adjacent = True        # omit label unless adjacent segment matches
adjacent_prob_factor = 0.65  # when checking if adjacent segment matches species, use self.min_prob times this
top_n = 6 # number of top matches to log in debug mode
min_location_freq = .0001    # ignore if species frequency less than this for location/week
file_date_regex = '\S+_(\d+)_.*' # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
file_date_regex_group = 1    # use group at offset 1
analyze_group_size = 100     # do this many files, then reset to avoid running out of GPU memory
frequency_db = 'frequency'   # eBird barchart data, i.e. species report frequencies

# Soundalike groups are used in analysis / inference when a location is given.
# For each soundalike species, eBird barchart data is accessed to get the maximum
# frequency across all weeks (i.e. maximum portion of checklists that include the species).
# If the maximum frequency for a species in a soundalike group is <= soundalike_cutoff,
# it is replaced by the species with the highest frequency > soundalike_cutoff in the group.
# For instance, if a Mountain Chickadee is ID'd in a county where it's never been seen,
# but Black-capped Chickadees are common there, it will be reported as a Black-capped Chickadee.
soundalike_cutoff = .005
soundalikes = [['Black-capped Chickadee', 'Boreal Chickadee', 'Mountain Chickadee'],
               ['Pacific Wren', 'Winter Wren'],
               ['Pine Warbler', 'Dark-eyed Junco']]

# paths
ignore_file = 'data/ignore.txt'     # classes listed in this file are ignored in analysis
classes_file = 'data/classes.txt'   # list of classes used in training and analysis
main_ckpt_name = 'ckpt_m'           # multi-label model checkpoint used in inference
search_ckpt_name = 'ckpt_s'         # single-label model checkpoint used to generate embeddings
