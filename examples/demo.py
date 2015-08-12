#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp
# import scipy.constants as spc
# import scipy.special as spp
import scisalt.matplotlib as sm
from blowout.Efield import E_complex
from blowout.Efield import E_gauss_circ as invlaw


# ======================================
# Set up 2x3 figure
# ======================================
fig, ax = sm.setup_axes(2, 3, figsize=(18, 12))

# ======================================
# RMS values for ellipse
# ======================================
sx = 10
sy = 1


# # ======================================
# # 2D Circular Gaussian fields
# # ======================================
# def invlaw(x, y, sr):
#     r = np.sqrt(x**2 + y**2)
#     E_mag = (1-np.exp( - r**2 / (2*sr**2))) / (2*np.pi*r)
#     E_x = E_mag * x / r  # noqa
#     E_y = E_mag * y / r  # noqa
#     return np.sqrt(E_x**2 + E_y**2)


# # ======================================
# # Bassetti-Erskine formula
# # ======================================
# def E_complex(x, y, sx, sy, q):
#     r_2_sx2_sy2 = np.sqrt(2*(sx**2 - sy**2))
#     r = sy/sx
#     a = x/r_2_sx2_sy2
#     b = y/r_2_sx2_sy2
#     aib = a + 1j*b
#     aribr = a*r + 1j*b/r
#     return - 1j / (2*np.sqrt(np.pi)*r_2_sx2_sy2) * (spp.wofz(aib) - np.exp(-aib**2 + aribr**2) * spp.wofz(aribr))

# ======================================
# Create grid
# ======================================
gridpts = 100
x = np.linspace(-5, 5, gridpts)
delx = x[1]-x[0]
y = np.linspace(-5, 5, gridpts)
dely = y[1]-y[0]
Xgrid, Ygrid = np.meshgrid(x, y)

# ======================================
# Evaluate Bassetti-Erskine
# ======================================
E = E_complex(Xgrid.T, Ygrid.T, sx, sy, 1)

# ======================================
# Get components
# ======================================
E_real = np.real(E)
E_imag = -np.imag(E)
E_mag = np.absolute(E)

# ======================================
# Evaluate 2D Circular Gaussian
# ======================================
new = np.abs(invlaw(Xgrid.T, Ygrid.T, 1))

# ======================================
# Plot Basetti-Erskine
# ======================================
im = sm.imshow(E_mag, ax=ax[0, 0], vmin=0, vmax=np.max(new))
sm.addlabel(ax=ax[0, 0], toplabel='Elliptical Field Magnitude')

# ======================================
# Plot 2D Circular Gaussian
# ======================================
im = sm.imshow(new, ax=ax[0, 1], vmin=0, vmax=np.max(new))
sm.addlabel(ax=ax[0, 1], toplabel='Circular Field Magnitude', xlabel='x', ylabel='y')

# ======================================
# Plot Field Strength Differences
# ======================================
im = sm.imshow(E_mag-new, ax=ax[0, 2], cmap='bwr', vmin=-0.04, vmax=0.04)
sm.addlabel(ax=ax[0, 2], toplabel='Field Magnitude Difference', xlabel='x', ylabel='y')

# ======================================
# Plot differences in major, minor axes
# ======================================
x = np.linspace(0, 10, 1000)
line1 = ax[1, 0].plot(x, invlaw(x, 0, 1), label='$E_x$, Circularly Symmetric')
# ylim = ax.get_ylim()
line2 = ax[1, 0].plot(x, np.abs(E_complex(x, 0, sx, sy, 1)), label='$E_x$, Elliptical')
line3 = ax[1, 0].plot(x, np.abs(E_complex(0, x, sx, sy, 1)), label='$E_y$, Elliptical')
# ax.set_ylim(ylim)

sm.addlabel(ax=ax[1, 0], toplabel='Ellipse vs. Circle', xlabel='r', ylabel='E Field')

ax[1, 0].legend(loc=1)

# ======================================
# Quiver
# ======================================
gridpts = 20
lims    = 10
x            = np.linspace(-lims, lims, gridpts)
delx         = x[1]-x[0]
y            = np.linspace(-lims, lims, gridpts)
dely         = y[1]-y[0]
Xgrid, Ygrid = np.meshgrid(x, y)
E            = E_complex(Xgrid.T, Ygrid.T, sx, sy, 1)
E_real       = np.real(E)
E_imag       = -np.imag(E)
E_mag        = np.absolute(E)
im           = sm.quiver(E_real, E_imag, ax=ax[1, 1])

# ======================================
# Working...
# ======================================
# fig, ax = sm.setup_axes()
x, y = sm.NonUniformImage_axes(new)

x = (x - np.mean(x)) * delx
y = (y - np.mean(y)) * dely

# sm.NonUniformImage(x, y, new, ax=ax)

plt.show()
