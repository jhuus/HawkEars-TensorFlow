# Plotting functions. Keep this separate from util.py since it imports libraries
# that most users don't need.

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from core import constants
    
# save a plot of a spectrogram
def plot_spec(spec, path, binary_classifier=False, gray_scale=False):
    if spec.ndim == 3:
        if binary_classifier:
            spec = spec.reshape((constants.BINARY_SPEC_HEIGHT, constants.SPEC_WIDTH))
        else:
            spec = spec.reshape((constants.SPEC_HEIGHT, constants.SPEC_WIDTH))

    plt.clf() # clear any existing plot data
    
    if gray_scale:
        spec = np.flipud(spec)
        spec = cv2.resize(spec, dsize=(384, 160), interpolation=cv2.INTER_CUBIC) # make it taller so it's easier to view
        plt.imshow(spec, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches = 0)
        plt.close()
    else:
        plt.pcolormesh(spec, shading='gouraud')
        plt.savefig(path)
        plt.close()
