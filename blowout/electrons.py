from .support import _timestamp2filename
import h5py as _h5
import numpy as _np
import pkg_resources as _pkg_resources
import scisalt as _ss
from .support import Timestamp as _Timestamp

_version = _pkg_resources.get_distribution('blowout').version


class PlasmaE(_Timestamp):
    """
    Base class for all generated plasmas.
    """
    def __init__(self, PlasmaParams, num_parts):
        super().__init__()
        self._PlasmaParams = PlasmaParams
        self._num_parts = num_parts


        # ======================================
        # Set up particle coordinates
        # ======================================
        steps = self.PlasmaParams.xi_bubble.size
        self.x_coords  = _np.empty(shape=(steps, num_parts))
        self.y_coords  = _np.empty(shape=(steps, num_parts))
        self.bx_coords = _np.empty(shape=(steps, num_parts))
        self.by_coords = _np.empty(shape=(steps, num_parts))

    @property
    def PlasmaParams(self):
        """
        The base plasma parameters.
        """
        return self._PlasmaParams

    @property
    def num_parts(self):
        """
        The number of particles used for the plasma electrons.
        """
        return self._num_parts

    def _set_timestamp(self, timestamp):
        self._timestamp = timestamp
        self._PlasmaParams._set_timestamp(timestamp)


    def write(self, filename=None):
        """
        Write all of the particles and plasma parameters to a file.
        """
        filename = _timestamp2filename(self, ftype='electrons', filename=filename)
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
            dx  = gdata.create_dataset(name='x_coords'  , data=self.x_coords  , shape=shape , compression="gzip")  # noqa
            dy  = gdata.create_dataset(name='y_coords'  , data=self.y_coords  , shape=shape , compression="gzip")  # noqa
            dbx = gdata.create_dataset(name='bx_coords' , data=self.bx_coords , shape=shape , compression="gzip")  # noqa
            dby = gdata.create_dataset(name='by_coords' , data=self.by_coords , shape=shape , compression="gzip")  # noqa

            gmeta = f.create_group('metadata')
            gmeta.attrs.create(name='num_parts' , data=self.num_parts )


class PlasmaE_Grid(PlasmaE):
    """
    Plasma electrons coordinates initialized in a `num_pts` by `num_pts` grid.
    """
    def __init__(self, num_pts, x_mag, y_mag, PlasmaParams):
        super().__init__(
            PlasmaParams = PlasmaParams,
            num_parts    = num_parts
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
    def __init__(self, num_parts, x_mag, y_mag, PlasmaParams):
        num_parts_quad = _np.int(num_parts/4)
        super().__init__(
            PlasmaParams = PlasmaParams,
            num_parts    = num_parts_quad * 4
            )
        self._x_mag   = x_mag
        self._y_mag   = y_mag

        # ======================================
        # Set initial conditions
        # ======================================
        x = _np.random.rand(num_parts_quad) * x_mag
        y = _np.random.rand(num_parts_quad) * y_mag
        self.x_coords[0, :] = _np.concatenate((x , -x , x  , -x))
        self.y_coords[0, :] = _np.concatenate((y , y  , -y , -y))
        self.bx_coords[0, :] = 0
        self.by_coords[0, :] = 0

    @property
    def num_parts(self):
        """
        The number of particles in the simulation.
        """
        return self._num_parts
