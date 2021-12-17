## Introduction
HawkEars is a desktop program that scans audio recordings for bird sounds and generates [Audacity](https://www.audacityteam.org/) label files. It is inspired by [BirdNET](https://github.com/kahst/BirdNET), and intended as an improved productivity tool for analyzing field recordings. This repository includes the source code and a trained model for a list of species found in northeastern North America. The complete list is found [here](https://github.com/jhuus/HawkEars/blob/main/data/classes.txt). The repository does not include the raw data or spectrograms used to train the model.

This project is licensed under the terms of the MIT license.

## Installation

The HawkEars neural network is implemented using [Tensorflow](https://www.tensorflow.org/), and requires a [CUDA-compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus), such as a Geforce RTX 10xx, Geforce RTX 20xx or Geforce RTX 30xx. It can be installed on Linux or Windows using the following steps: 

1.	Install [Python 3](https://www.python.org/downloads/), if you do not already have it installed.
2.	Download a copy of this repository. If you have Git installed, type:

```
 git clone https://github.com/jhuus/HawkEars
```
 
Otherwise you can click on the Code link at the top, select “Download ZIP” and unzip it after it’s been downloaded.

3.	Install required Python libraries:

```
pip install -r requirements.txt
```

4.	Install ffmpeg. On Linux, type:

```
sudo apt-get install ffmpeg
```

On Windows, see https://www.ffmpeg.org/download.html#build-windows 

5. Install [CUDA](https://docs.nvidia.com/cuda/)

## Analyzing Field Recordings
To run analysis, type:

```
python analyze.py -i <input path> -o <output path> 
```

The input path can be a directory or a reference to a single audio file, but the output path must be a directory, where the generated Audacity label files will be stored. As a quick first test, try:

```
python analyze.py -i test -o test
```

This will analyze the recording(s) included in the test directory. There are also a number of optional arguments, which you can review by typing: 

```
python analyze.py -h
```

If you don't have access to additional recordings for testing, one good source is [xeno-canto](https://xeno-canto.org/). Recordings there are generally single-species, however, and therefore somewhat limited. A source of true field recordings, generally with multiple species, is the [Hamilton Bioacoustics Field Recordings](https://archive.org/details/hamiltonbioacousticsfieldrecordings).

After running analysis, you can view the output by opening an audio file in Audacity, clicking File / Import / Labels and selecting the generated label file. Audacity should then look something like this:

![](audacity-labels.png)

To show spectrograms by default in Audacity, click Edit / Preferences / Tracks and set Default View Mode = Spectrogram. You can modify the spectrogram settings under Edit / Preferences / Tracks / Spectrograms.

You can click a label to view or listen to that segment. All segments are currently 3 seconds long, although there is an option to merge matching adjacent labels. You can also edit a label if a bird is misidentified, or delete and insert labels, and then export the edited label file for use as input to another process. Label files are simple text files and easy to process for data analysis purposes.

## Limitations
Some bird species are difficult to identify by sound alone. This includes mimics, for obvious reasons, which is why Northern Mockingbird is not currently included in the species list. European Starlings are included, but often mimic other birds and are therefore sometimes challenging to identify. Species that have been excluded because they sound too much like other species are Boreal Chickadee (sounds like Black-capped Chickadee), Hoary Redpoll (sounds like Common Redpoll) and Philadelphia Vireo (sounds like Red-eyed Vireo). 

Identification is based on 3-second segments, and for some species (e.g. Brown Thrasher) a longer segment would be better. I plan to explore that more in future. Also, sometimes a segment includes just the beginning or end of a song, which can lead to misidentification. 

Currently, HawkEars identifies exactly one class (species or non-bird sound) per segment. In practice of course it's common to hear two or more birds at once. That can lead to misidentification, and is something I plan to explore in future. There are well-known image classification techniques for handling this.

The current list is missing a number of species, especially shorebirds and waterfowl, so adding species is another future task. Also, it's missing sound types for many species. For example, flight calls and juvenile begging sounds are mostly missing. That's partly due to difficulty getting enough good recordings of these sounds, but it's certainly an area for further work. 

## Preparing to Train Your Own Model
to do

## Training Your Own Model
to do

## Implementation Notes
### Neural Network
The neural network used by HawkEars is based on the ResNeST design introduced in [this paper](https://arxiv.org/pdf/2004.08955.pdf). It is an image-classification network, trained on spectrogram images. The implementation is in model/resnest.py, which defines a ResNest class. Key parameters for sizing the model are num_stages and blocks_set. A ResNeST model consists of a sequence of stages, each of which contains one or more blocks. For example, setting num_stages=3 and blocks_set=[3,2,1] defines a model with three stages, where the first stage has 3 blocks, the second has 2 blocks and the third has 1 block. When running train.py, these are set using the -s parameter for stages, -n1 for blocks in the first stage, -n2 for blocks in the second stage, etc.   

### Spectrograms
Spectrograms are extracted from audio files in core/audio.py, then compressed and saved to a SQLite database. (more content required here).
