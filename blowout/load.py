from .drive import Drive
from .electrons import PlasmaE
from .simframework import SimFrame
from .plasma import PlasmaParams
from .ions import PlasmaIons
import h5py as _h5
import logging as _logging
import pkg_resources as _pkg_resources
from .support import _read_arrays
from .support import _read_dict

_logger  = _logging.getLogger(__name__)
_version = _pkg_resources.get_distribution('blowout').version


def loadSim(filebase):
    """
    Loads simulation.
    """
    params    = loadPlasmaParams(filename='{}.plasmaparams.h5'.format(filebase))
    electrons = loadPlasmaE(plasmaparams=params, filename='{}.electrons.h5'.format(filebase))
    drive     = loadDrive(filename='{}.drive.h5'.format(filebase))
    ions      = loadPlasmaIons(plasmaparams=params, filename='{}.ions.h5'.format(filebase))
    
    return SimFrame(
        Drive      = drive,
        PlasmaE    = electrons,
        PlasmaIons = ions
        )


def loadPlasmaParams(filename=None, gui=True):
    """
    Load plasma parameters.

    Returns :class:`blowout.plasma.PlasmaParams`.
    """
    if filename is None and gui:
        import scisalt.qt as _ssqt
        filename = _ssqt.getOpenFileName()

    with _h5.File(name=filename, mode='r') as f:
        _checkversion(f)

        # ======================================
        # Load metadata
        # ======================================
        mattrs = f['metadata'].attrs
        xi_start  = mattrs['xi_start']
        xi_end    = mattrs['xi_end']
        dxi       = mattrs['dxi']
        np        = mattrs['np']

    # ======================================
    # Create class
    # ======================================
    plasmaparams = PlasmaParams(
        xi_start = xi_start,
        xi_end   = xi_end,
        dxi      = dxi,
        np       = np
        )

    return plasmaparams


def loadPlasmaIons(plasmaparams=None, filename=None, gui=True):
    """
    Load plasma ions.

    Returns :class:`blowout.ions.PlasmaIons`.
    """
    if filename is None and gui:
        import scisalt.qt as _ssqt
        filename = _ssqt.getOpenFileName()

    with _h5.File(name=filename, mode='r') as f:
        _checkversion(f)

        # ======================================
        # Load metadata
        # ======================================
        mattrs = f['metadata'].attrs

        # ======================================
        # Load data
        # ======================================
        data = f['data']

        img            = _read_arrays(data, 'img')
        bounds         = _read_arrays(data, 'bounds')
        closed_ellipse = _read_arrays(data, 'closed_ellipse')
        xind           = data['xind'].value
        yind           = data['yind'].value
        results_flat = _read_dict(data, 'results_flat')

    # ======================================
    # Create class
    # ======================================
    plas = PlasmaIons(
        PlasmaParams = plasmaparams,
        )

    # ======================================
    # Put data into class
    # ======================================
    plas._img            = img
    plas._bounds         = bounds
    plas._closed_ellipse = closed_ellipse
    plas._xind           = xind
    plas._yind           = yind
    plas._results_flat   = results_flat

    # # ======================================
    # # Load coordinate data
    # # ======================================
    # plas.x_coords  = x_coords
    # plas.y_coords  = y_coords
    # plas.bx_coords = bx_coords
    # plas.by_coords = by_coords

    return plas


def loadPlasmaE(plasmaparams=None, filename=None, gui=True):
    """
    Load plasma particles.

    Returns :class:`blowout.electrons.PlasmaE`.
    """
    if filename is None and gui:
        import scisalt.qt as _ssqt
        filename = _ssqt.getOpenFileName()

    with _h5.File(name=filename, mode='r') as f:
        _checkversion(f)

        # ======================================
        # Load metadata
        # ======================================
        mattrs = f['metadata'].attrs
        num_parts = mattrs['num_parts']

        # ======================================
        # Load data
        # ======================================
        data = f['data']

        x_coords  = data['x_coords'].value
        y_coords  = data['y_coords'].value
        bx_coords = data['bx_coords'].value
        by_coords = data['by_coords'].value

    # ======================================
    # Create class
    # ======================================
    plas = PlasmaE(
        PlasmaParams = plasmaparams,
        num_parts    = num_parts,
        )

    # ======================================
    # Load coordinate data
    # ======================================
    plas.x_coords  = x_coords
    plas.y_coords  = y_coords
    plas.bx_coords = bx_coords
    plas.by_coords = by_coords

    return plas


def loadDrive(filename=None, gui=True):
    if filename is None and gui:
        import scisalt.qt as _ssqt
        filename = _ssqt.getOpenFileName()

    with _h5.File(name=filename, mode='r') as f:
        _checkversion(f)

        # ======================================
        # Load metadata
        # ======================================
        mattrs = f['metadata'].attrs
        sx     = mattrs['sx']
        sy     = mattrs['sy']
        sz     = mattrs['sz']
        charge = mattrs['charge']
        gamma  = mattrs['gamma']

    return Drive(
        sx     = sx,
        sy     = sy,
        sz     = sz,
        charge = charge,
        gamma  = gamma
        )
    

def _checkversion(f):
    # ======================================
    # Check version
    # ======================================
    fversion = f.attrs['version']
    if fversion != _version:
        _logger.critical('Versions do not match! File: {}; Software: {}'.format(fversion, _version))
