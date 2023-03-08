This folder contains two checkpoints based on the following audio parameters:

segment_len = 3
mel_scale = True
hop_length = 320
win_length = 2048
spec_height = 128
spec_width = 384
min_audio_freq = 200
max_audio_freq = 10500

Notably, height=128, unlike the main model which uses 256.
ckpt_m is a multi-label model intended for use in transfer learning.
ckpt_s is a single-label model used to generate embeddings.
Note that neither was generated from the current class list.
