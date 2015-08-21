from . import Efield as _Efield
from .simframework import SimFrame
import h5py as _h5
import logging as _logging
import numpy as _np
import pkg_resources as _pkg_resources
import scipy.constants as _spc
import scisalt as _ss
import time as _time

# import ipdb
_version = _pkg_resources.get_distribution('blowout').version

_logger = _logging.getLogger(__name__)


def loadSim(filebase):
    """
    Loads simulation.
    """
    plas  = loadPlasma(filename='{}.plasma.h5'.format(filebase))
    drive = loadDrive(filename='{}.drive.h5'.format(filebase))
    
    return SimFrame(drive, plas)


def loadPlasma(filename=None, gui=True):
    """
    Load plasma particles.

    Returns :class:`blowout.generate.PlasmaE`.
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
        num_parts = mattrs['num_parts']
        np        = mattrs['np']

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
        xi_start  = xi_start,
        xi_end    = xi_end,
        dxi       = dxi,
        num_parts = num_parts,
        np        = np
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
    

class Drive(object):
    """
    Contains properties derived from the drive bunch.
    """
    def __init__(self, sx, sy, sz, charge, gamma):
        self._sx        = sx
        self._sy        = sy
        self._sz        = sz
        self._charge    = charge
        self._gamma     = gamma
        self._timestamp = None

    @property
    def timestamp(self):
        if self._timestamp is not None:
            return self._timestamp
        else:
            raise RuntimeError('No timestamp: simulation not completed.')

    @property
    def sx(self):
        """
        Drive beam Gaussian standard deviation in :math:`x`.
        """
        return self._sx

    @property
    def sy(self):
        """
        Drive beam Gaussian standard deviation in :math:`y`.
        """
        return self._sy

    @property
    def sz(self):
        """
        Drive beam Gaussian standard deviation in :math:`z`.
        """
        return self._sz

    @property
    def charge(self):
        """
        Drive beam total charge in Coulombs.
        """
        return self._charge

    @property
    def gamma(self):
        """
        Relativistic gamma :math:`\\gamma` of bunch.
        """
        return self._gamma

    def E_fields(self, x, y, xi):
        """
        Returns the fields at :math:`(x, y)`.
        """
        # This actually is q = rho(z) * dz / dz.
        q = self.charge * _ss.numpy.gaussian(xi, 0, self.sz)
        return _Efield.E_complex(x, y, self.sx, self.sy, self.charge)

    def write(self, filename=None):
        filename = _timestamp2filename(self, ftype='drive', filename=filename)
        # ======================================
        # Create new filename
        # ======================================
        with _h5.File(filename, 'w') as f:
            # ipdb.set_trace()
            f.attrs['version'] = _version
            # f.attrs.create(name='version', data=_version)

            # ======================================
            # Write metadata
            # ======================================
            gmeta = f.create_group('metadata')
            gmeta.attrs.create(name='sx'     , data=self.sx     )
            gmeta.attrs.create(name='sy'     , data=self.sy     )
            gmeta.attrs.create(name='sz'     , data=self.sz     )
            gmeta.attrs.create(name='charge' , data=self.charge )
            gmeta.attrs.create(name='gamma'  , data=self.gamma  )


class PlasmaE(object):
    """
    Base class for all generated plasmas.
    """
    def __init__(self, xi_start, xi_end, dxi, num_parts, np):
        self._xi_start  = xi_start
        self._xi_end    = xi_end
        self._dxi       = dxi
        self._num_parts = num_parts
        self._np        = np
        self._dt        = self._dxi / _spc.speed_of_light
        self._timestamp = None

        xi_bubble       = _ss.numpy.linspacestep(self.xi_start, self.xi_end, self.dxi)
        # self._xi_bubble = xi_bubble[0:5]
        self._xi_bubble = xi_bubble

        # ======================================
        # Set up particle coordinates
        # ======================================
        self.x_coords  = _np.empty(shape=(self._xi_bubble.size, num_parts))
        self.y_coords  = _np.empty(shape=(self._xi_bubble.size, num_parts))
        self.bx_coords = _np.empty(shape=(self._xi_bubble.size, num_parts))
        self.by_coords = _np.empty(shape=(self._xi_bubble.size, num_parts))

    @property
    def np(self):
        """
        Plasma density :math:`n_p`.
        """
        return self._np

    @property
    def xi_bubble(self):
        """
        Vector of individual slices in :math:`\\xi`.
        """
        return self._xi_bubble

    @property
    def xi_start(self):
        """
        Maximum simulation value for starting value of :math:`\\xi`.
        """
        return self._xi_start

    @property
    def xi_end(self):
        """
        Maximum simulation value for ending value of :math:`\\xi`.
        """
        return self._xi_end

    @property
    def dxi(self):
        """
        Simulation step :math:`\\Delta \\xi`.
        """
        return self._dxi

    @property
    def dt(self):
        """
        Simulation step :math:`\\Delta t`.
        """
        return self._dt

    @property
    def timestamp(self):
        if self._timestamp is not None:
            return self._timestamp
        else:
            raise RuntimeError('No timestamp: simulation not completed.')

    def write(self, filename=None):
        """
        Write all of the particles and plasma parameters to a file.
        """
        filename = _timestamp2filename(self, ftype='plasma', filename=filename)
        # ======================================
        # Create new filename
        # ======================================
        with _h5.File(filename, 'w') as f:
            # ipdb.set_trace()
            f.attrs['version'] = _version
            # f.attrs.create(name='version', data=_version)
            
            shape = self.x_coords.shape
            gdata = f.create_group('data')

            # ======================================
            # Write data
            # ======================================
            dx  = gdata.create_dataset(name='x_coords'  , data=self.x_coords  , shape=shape , compression="gzip")
            dy  = gdata.create_dataset(name='y_coords'  , data=self.y_coords  , shape=shape , compression="gzip")
            dbx = gdata.create_dataset(name='bx_coords' , data=self.bx_coords , shape=shape , compression="gzip")
            dby = gdata.create_dataset(name='by_coords' , data=self.by_coords , shape=shape , compression="gzip")

            # ======================================
            # Write metadata
            # ======================================
            gmeta = f.create_group('metadata')
            gmeta.attrs.create(name='xi_start'  , data=self.xi_start  )
            gmeta.attrs.create(name='xi_end'    , data=self.xi_end    )
            gmeta.attrs.create(name='dxi'       , data=self.dxi       )
            gmeta.attrs.create(name='num_parts' , data=self.num_parts )
            gmeta.attrs.create(name='np'        , data=self.np        )


class PlasmaE_Grid(PlasmaE):
    """
    Plasma electrons coordinates initialized in a `num_pts` by `num_pts` grid.
    """
    def __init__(self, num_pts, x_mag, y_mag, xi_start, xi_end, dxi, np):
        super().__init__(
            xi_start  = xi_start,
            xi_end    = xi_end,
            dxi       = dxi,
            num_parts = num_pts*num_pts,
            np        = np
            )
        self._num_pts = num_pts
        self._x_mag   = x_mag
        self._y_mag   = y_mag

        # ======================================
        # Set up transverse grid
        # ======================================
        # num_pts = 100
        # x_mag = 150e-6
        # y_mag = 150e-6

        x_vec = _np.linspace(-x_mag, x_mag, num_pts)
        y_vec = _np.linspace(-y_mag, y_mag, num_pts)
        
        # ======================================
        # Set initial conditions
        # ======================================
        x_c, y_c = _np.meshgrid(x_vec, y_vec)
        self.x_coords[0, :] = x_c.flatten()
        self.y_coords[0, :] = y_c.flatten()
        self.bx_coords[0, :] = 0
        self.by_coords[0, :] = 0

    @property
    def num_pts(self):
        """
        The number of transverse points in the grid.
        """
        return self._num_pts

    @property
    def num_parts(self):
        """
        The number of particles in the simulation.
        """
        return self._num_pts**2


class PlasmaE_Random(PlasmaE):
    """
    Plasma electrons coordinates for `num_parts` particles, initialized randomly within a box with :math:`-x_{mag} < x < x_{mag}` and :math:`-y_{mag} < y < y_{mag}`.
    """
    def __init__(self, num_parts, x_mag, y_mag, xi_start, xi_end, dxi, np):
        super().__init__(
            xi_start  = xi_start,
            xi_end    = xi_end,
            dxi       = dxi,
            num_parts = num_parts,
            np        = np
            )
        self._x_mag   = x_mag
        self._y_mag   = y_mag

        # ======================================
        # Set initial conditions
        # ======================================
        self.x_coords[0, :] = _np.random.rand(num_parts) * (2 * x_mag) - x_mag
        self.y_coords[0, :] = _np.random.rand(num_parts) * (2 * y_mag) - y_mag
        self.bx_coords[0, :] = 0
        self.by_coords[0, :] = 0

    @property
    def num_parts(self):
        """
        The number of particles in the simulation.
        """
        return self._num_parts


def _timestamp2filename(cls, ftype, filename=None):
    # ======================================
    # Get filename from timestamp
    # ======================================
    if filename is None:
        try:
            timestamp = cls.timestamp
        except RuntimeError as err:
            _logger.debug('Handled exception: {}'.format(err))
            timestamp = _time.localtime()

        filename = _time.strftime('%Y.%m.%d.%H%M.%S.{}.h5'.format(ftype), timestamp)

    return filename

def _checkversion(f):
    # ======================================
    # Check version
    # ======================================
    fversion = f.attrs['version']
    if  fversion != _version:
        _logger.critical('Versions do not match! File: {}; Software: {}'.format(fversion, _version))
