#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import blowout as bo
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as spc
import scisalt as ss
import scisalt.matplotlib as sm
import skimage.feature as skfeat
import skimage.measure as skmeas
import logging as _logging
import sys

# import ipdb

level = _logging.DEBUG
logger = ss.logging.mylogger(filename='run')

# ======================================
# Get relevant constants
# ======================================
e         = spc.elementary_charge
sy        = 2e-6
sx        = sy*4
sz        = 30e-6
qtot      = 2e10*e
mag       = 1e-8
xi_start  = -5*sz
xi_end    = 0
dxi       = sz/5
# dxi       = sz
num_parts = 1e5
npl        = 1e18

num_pts   = 100
shape = (num_pts, num_pts)

Drive = bo.drive.Drive(sx, sy, sz=sz, charge=qtot, gamma=39824)

PlasmaParams = bo.plasma.PlasmaParams(
    xi_start = xi_start,
    xi_end   = xi_end,
    dxi      = dxi,
    np       = npl
    )

PlasmaE = bo.electrons.PlasmaE_Random(
    x_mag        = mag,
    y_mag        = mag,
    num_parts    = num_parts,
    PlasmaParams = PlasmaParams
    )

PlasmaIons = bo.ions.PlasmaIons(
    PlasmaParams = PlasmaParams
    )

sim = bo.SimFrame(Drive=Drive, PlasmaE=PlasmaE, PlasmaIons=PlasmaIons)
sim.sim()
# sim.write(filename='test')
sim.write()
