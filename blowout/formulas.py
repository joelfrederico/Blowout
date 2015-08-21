import numpy as _np
from .Efield import E_complex as _E_complex
import scipy.constants as _spc

__all__ = [
    'gamma',
    'a'
    ]
__all__.sort()


# ======================================
# Return relativistic gamma
# ======================================
def gamma(bx, by):
    """
    Returns the relativistic factor :math:`\\gamma` given normalized velocities :math:`\\beta_x` and :math:`\\beta_y`.
    """
    return _np.power(1 - bx**2 - by**2, -1/2)


# ======================================
# Get acceleration
# ======================================
def a(x, y, bx, by, E_x, E_y):
    """
    Acceleration of an electron with position :math:`(x, y)` and normalized velocity :math:`(\\beta_x, \\beta_y)` given an electric field :math:`\\vec{E} = E_x \\hat{x} + E_y \\hat{y}`.
    """
    # ======================================
    # Useful substitutions
    # ======================================
    g   = gamma(bx, by)
    g2  = g**2
    gme = g * _spc.electron_mass
    e   = _spc.elementary_charge

    ax  = e * E_x / (gme) - (g2 * bx * by * e * E_y / ((by**2 * g2 + 1) * gme))
    ay  = (e * E_y / (gme) - g2*bx*by*ax) / (by**2 * g2 + 1)

    return ax, ay


# ======================================
# Beta derivatives
# ======================================
def dbetadt(x, y, bx, by, Ex, Ey):
    """
    Derivative of beta in x and y (:math:`\\frac{d}{dt}\\vec{\\beta}` of an electron with position :math:`(x, y)` and normalized velocity :math:`(\\beta_x, \\beta_y)` given an electric field :math:`\\vec{E} = E_x \\hat{x} + E_y \\hat{y}`.
    """
    g = gamma(bx, by)
    g2inv = _np.power(g, -2.0)
    gmc = g*_spc.electron_mass*_spc.speed_of_light
    e = _spc.elementary_charge

    dbxdt = e/gmc*(Ex*(by**2 + g2inv) - Ey*bx*by)
    dbydt = e/gmc*(Ey*(bx**2 + g2inv) - Ex*bx*by)

    return dbxdt, dbydt
