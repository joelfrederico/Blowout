import numpy as _np
import scipy.special as _spp
import scipy.constants as _spc
import pdb

__all__ = [
    'E_complex',
    'E_gauss_circ'
    ]


# ======================================
# Bassetti-Erskine formula
# ======================================
def E_complex(x, y, sx, sy, q):
    """
    The fields at :math:`(x, y)` of an elliptical gaussian region of charge with standard deviations ``sx`` and ``sq``, and total charge ``q``.

    Returns ``E_x, E_y``.
    """
    r_2_sx2_sy2 = _np.sqrt(2*(sx**2 - sy**2))
    r = sy/sx
    a = _np.abs(x)/r_2_sx2_sy2
    b = _np.abs(y)/r_2_sx2_sy2
    aib = a + 1j*b
    aribr = a*r + 1j*b/r
    # pdb.set_trace()
    E_c = - 1j*q / (2*_spc.epsilon_0*_np.sqrt(_np.pi)*r_2_sx2_sy2) * (_spp.wofz(aib) - _np.exp(-aib**2 + aribr**2) * _spp.wofz(aribr))
    E_x = _np.real(E_c) * _np.sign(x)
    E_y = -_np.imag(E_c) * _np.sign(y)
    return E_x, E_y


def E_x(x, y, sx, sy, q):
    return _np.real(E_complex(x, y, sx, sy, q))


def E_y(x, y, sx, sy, q):
    return _np.imag(E_complex(x, y, sx, sy, q))


# ======================================
# Circular
# ======================================
def E_gauss_circ(x, y, sr):
    r = _np.sqrt(x**2 + y**2)
    E_mag = (1-_np.exp( - r**2 / (2*sr**2))) / (2*_np.pi*r)
    E_x = E_mag * x / r  # noqa
    E_y = E_mag * y / r  # noqa
    return _np.sqrt(E_x**2 + E_y**2)
