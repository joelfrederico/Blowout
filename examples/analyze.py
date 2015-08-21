#!/usr/bin/env python3

import blowout as bo
import scisalt as ss
import matplotlib.pyplot as plt
import numpy as np
import skimage.feature as skfeat


sim = bo.generate.loadSim('2015.08.20.1825.34')
# plas = bo.generate.loadPlasma(filename)
PlasmaE = sim.PlasmaE
Drive = sim.Drive

x_coords  = PlasmaE.x_coords[-2:-1]
y_coords  = PlasmaE.y_coords[-2:-1]
bx_coords = PlasmaE.bx_coords[-2:-1]
by_coords = PlasmaE.by_coords[-2:-1]

# ipdb.set_trace()

plt.ion()

ind = np.random.choice(np.arange(0, len(x_coords[0])), size=1000, replace=False)


def plot_step(x, y, bx, by):
    # g   = bo.formulas.gamma(bx, by)
    # img = g.reshape(shape)
    # ss.matplotlib.Imshow_Slider(img)

    fig, ax = ss.matplotlib.setup_axes(rows=1, cols=2)

    ss.matplotlib.hist2d(x, y, bins=50, ax=ax[0, 0])
    ss.matplotlib.quiver(x[ind], y[ind], bx[ind], by[ind], ax=ax[0, 1])

    plt.show()

for x, y, bx, by in zip(x_coords, y_coords, bx_coords, by_coords):
    pass
    # plot_step(x, y, bx, by)

# ======================================
# Find evacuated ellipse
# ======================================
ind = np.abs(x) < 3

img, discard = ss.matplotlib.hist2d(x[ind], y[ind], bins=200, plot=False)
ss.matplotlib.Imshow_Slider(img)
# img_thresh = img < 1
# labels = skmeas.label(img_thresh)
edges = skfeat.canny(img)


# ss.matplotlib.Imshow_Slider(edges)

plt.ion()
