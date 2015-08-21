import scisalt as ss
import scipy.constants as _spc
import numpy as _np
from .formulas import dbetadt as _a
import time as _time
import logging as _logging
_logger = _logging.getLogger(__name__)


__all__ = [
    'SimFrame'
    ]


class SimFrame(object):
    """
    Coordinates and steps through the simulation.
    """
    def __init__(self, Drive, PlasmaE):
        self._Drive     = Drive
        self._PlasmaE   = PlasmaE
        self._timestamp = None

    def sim(self):
        # ======================================
        # Set up longitudinal coord
        # ======================================
        dt = self.PlasmaE.dxi / _spc.speed_of_light
        _logger.info('Time step dt: {}'.format(dt))

        PlasmaE = self.PlasmaE
        
        with ss.utils.progressbar(total=len(PlasmaE.xi_bubble), length=100) as myprog:
            for i, xi in enumerate(PlasmaE.xi_bubble[0:-1]):
                myprog.step = i+1
                PlasmaE.x_coords[i+1, :] = PlasmaE.x_coords[i, :] + PlasmaE.bx_coords[i, :] * _spc.speed_of_light * dt * 1e11
                PlasmaE.y_coords[i+1, :] = PlasmaE.y_coords[i, :] + PlasmaE.by_coords[i, :] * _spc.speed_of_light * dt * 1e11
                for j, (x, y, bx, by) in enumerate(zip(PlasmaE.x_coords[i, :], PlasmaE.y_coords[i, :], PlasmaE.bx_coords[i, :], PlasmaE.by_coords[i, :])):
                    E = self.Drive.E_fields(x, y, xi) * self.Drive.gamma
                    acc = _a(x, y, bx, by, E[0], E[1])
                    PlasmaE.bx_coords[i+1, j] = PlasmaE.bx_coords[i, j] + acc[0] * dt
                    PlasmaE.by_coords[i+1, j] = PlasmaE.by_coords[i, j] + acc[1] * dt

        # ======================================
        # Record completion timestamp
        # ======================================
        self._timestamp         = _time.localtime()
        self.PlasmaE._timestamp = self._timestamp
        self.Drive._timestamp   = self._timestamp

    @property
    def PlasmaE(self):
        """
        The plasma :class:`blowout.generate.PlasmaE` used for the simulation.
        """
        return self._PlasmaE

    @property
    def Drive(self):
        """
        The drive :class:`blowout.generate.Drive` used for the simulation.
        """
        return self._Drive

    def write(self, filename=None):
        self.PlasmaE.write(filename=filename)
        self.Drive.write(filename=filename)
