#!/usr/bin/env python3

import blowout as bo
import matplotlib.pyplot as plt
import numpy as np
import pdb
import scisalt as ss
import scisalt.matplotlib.Imshow_Slider as Imshow_Slider
import skimage.draw as skdraw
import skimage.feature as skfeat
import skimage.measure as skmeas
import skimage.morphology as skmorph
import skimage.segmentation as skseg
import skimage.transform as sktrans
import time


# filename = 'test'
filename = '2015.08.27.1654.16'
sim      = bo.load.loadSim(filename)

plt.ion()
# sim.PlasmaIons.draw_ellipse(i=1)

def plotimg(i):
    Imshow_Slider(sim.PlasmaIons._img[i])

# num_samples = len(x_coords[0])
# num_subset = 1000
# ind = np.arange(0, num_samples)
# if num_samples > num_subset:
#     ind = np.random.choice(ind, size=num_subset, replace=False)
