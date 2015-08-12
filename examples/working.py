#!/usr/bin/env python3

import numpy as np
import scisalt as ss
import blowout as bl
import scipy.constants as spc

# ======================================
# Get relevant constants
# ======================================
e  = spc.elementary_charge
me = spc.electron_mass
sy = 2e-6
sx = sy*4
qtot = 2e10*e

# ======================================
# Set up transverse grid
# ======================================
num_pts = 100
x_mag = 150e-6
y_mag = 150e-6
x_vec = np.linspace(-x_mag, x_mag, num_pts)
y_vec = np.linspace(-y_mag, y_mag, num_pts)

# ======================================
# Set up longitudinal coord
# ======================================
dxsi = 100e-9
dt = dxsi / spc.speed_of_light
xsi_max = 100e-6
xsi_bubble = ss.numpy.linspacestep(0, xsi_max, dxsi)
xsi_bubble = xsi_bubble[0:2]

# ======================================
# Set up particle coordinates
# ======================================
x_coords = np.empty(shape=(xsi_bubble.size, num_pts * num_pts))
y_coords = np.empty(shape=(xsi_bubble.size, num_pts * num_pts))
bx_coords = np.empty(shape=(xsi_bubble.size, num_pts * num_pts))
by_coords = np.empty(shape=(xsi_bubble.size, num_pts * num_pts))

# ======================================
# Set initial conditions
# ======================================
x_c, y_c = np.meshgrid(x_vec, y_vec)
x_coords[0, :] = x_c.flatten()
y_coords[0, :] = y_c.flatten()
bx_coords[0, :] = 0
by_coords[0, :] = 0


# ======================================
# Return relativistic gamma
# ======================================
def gamma(bx, by):
    return np.power(1 - bx**2 - by**2, -1/2)


# ======================================
# Get acceleration
# ======================================
def a(x, y, bx, by):
    # ======================================
    # Get e field
    # ======================================
    E_c    = bl.Efield.E_complex(x, y, sx, sy, qtot)
    E_x = np.real(E_c)
    E_y = np.imag(E_c)
    g = gamma(bx, by)
    g2 = g**2
    gme = g * me
    ax = e * E_x / (gme) - (g2 * bx * by * e * E_y / ((by**2 * g2 + 1) * gme))
    ay = (e * E_y / (gme) - g2*bx*by*ax) / (by**2 * g2 + 1)
    return ax, ay

for i, xsi in enumerate(xsi_bubble[0:-1]):
    x_coords[i+1, :] = x_coords[i, :] + bx_coords[i, :] * spc.speed_of_light
    y_coords[i+1, :] = y_coords[i, :] + by_coords[i, :] * spc.speed_of_light
    for j, (x, y, bx, by) in enumerate(zip(x_coords[i, :], y_coords[i, :], bx_coords[i, :], by_coords[i, :])):
        acc = a(x, y, bx, by)
        bx_coords[i+1, j] = bx_coords[i, j] + acc[0] * dt
        by_coords[i+1, j] = by_coords[i, j] + acc[1] * dt
