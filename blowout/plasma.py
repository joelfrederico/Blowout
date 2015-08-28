from .support import _timestamp2filename
import h5py as _h5
import pkg_resources as _pkg_resources
import scipy.constants as _spc
import scisalt as _ss

_version = _pkg_resources.get_distribution('blowout').version


class PlasmaParams(object):
    """
    Base plasma parameters.
    """
    def __init__(self, xi_start, xi_end, dxi, np):
        self._xi_start  = xi_start
        self._xi_end    = xi_end
        self._dxi       = dxi
        self._np        = np

        self._xi_bubble = _ss.numpy.linspacestep(self.xi_start, self.xi_end, self.dxi)
        self._dt        = self._dxi / _spc.speed_of_light

        self._timestamp = None

    def _set_timestamp(self, timestamp):
        self._timestamp = timestamp

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def num_steps(self):
        """
        Number of steps in sim.
        """
        return self._xi_bubble.size

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

    def write(self, filename=None):
        """
        Write all of the plasma parameters to a file.
        """
        filename = _timestamp2filename(self, ftype='plasmaparams', filename=filename)
        # ======================================
        # Create new filename
        # ======================================
        with _h5.File(filename, 'w') as f:
            f.attrs['version'] = _version
            
            # ======================================
            # Write metadata
            # ======================================
            gmeta = f.create_group('metadata')
            gmeta.attrs.create(name='xi_start'  , data=self.xi_start  )
            gmeta.attrs.create(name='xi_end'    , data=self.xi_end    )
            gmeta.attrs.create(name='dxi'       , data=self.dxi       )
            gmeta.attrs.create(name='np'        , data=self.np        )
