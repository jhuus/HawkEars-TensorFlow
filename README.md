## Introduction
HawkEars is a desktop program that scans audio recordings for bird sounds and generates [Audacity](https://www.audacityteam.org/) label files. It is inspired by [BirdNET](https://github.com/kahst/BirdNET), and intended as an improved productivity tool for processing field recordings. This repository includes the source code and a trained model for a list of species found in northeastern North America. The complete list is found [here](https://github.com/jhuus/HawkEars/blob/main/data/classes.txt). The repository does not include the raw data or spectrograms used to train the model.

This project is licensed under the terms of the MIT license.

## Installation
1.	Install [Python](https://www.python.org/downloads/), if you do not already have it installed.
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

5. If you have a compatible NVIDIA GPU, you can get improved performance by installing [CUDA](https://docs.nvidia.com/cuda/). You may want to test without CUDA first though, to ensure your basic setup is correct and to collect baseline performance numbers. 

## Analyzing Field Recordings
TBD

## Preparing to Train Your Own Model
TBD

## Training Your Own Model
TBD
