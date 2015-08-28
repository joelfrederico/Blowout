from . import Efield as _Efield
import h5py as _h5
import pkg_resources as _pkg_resources
import scisalt as _ss
from .support import _timestamp2filename
from .support import Timestamp as _Timestamp

_version = _pkg_resources.get_distribution('blowout').version


class Drive(_Timestamp):
    """
    Contains properties derived from the drive bunch.
    """
    def __init__(self, sx, sy, sz, charge, gamma):
        super().__init__()
        self._sx        = sx
        self._sy        = sy
        self._sz        = sz
        self._charge    = charge
        self._gamma     = gamma
        self._timestamp = None

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
        return _Efield.E_complex(x, y, self.sx, self.sy, q)

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
