from .formulas import dbetadt as _a
import logging as _logging
import numpy as _np
import scipy.constants as _spc
import scisalt as _ss
import time as _time
_logger = _logging.getLogger(__name__)


__all__ = [
    'SimFrame'
    ]


class SimFrame(object):
    """
    Coordinates and steps through the simulation.
    """
    def __init__(self, Drive, PlasmaE, PlasmaIons):
        self._Drive      = Drive
        self._PlasmaE    = PlasmaE
        self._PlasmaIons = PlasmaIons
        self._timestamp  = None

    def sim(self):
        # ======================================
        # Set up longitudinal coord
        # ======================================
        # dt = self.PlasmaE.dxi / _spc.speed_of_light
        dt = self.PlasmaE.PlasmaParams.dt
        _logger.info('Time step dt: {}'.format(dt))

        PlasmaE = self.PlasmaE
        
        # ======================================
        # Push particles
        # ======================================
        with _ss.utils.progressbar(total=len(PlasmaE.PlasmaParams.xi_bubble), length=100) as myprog:
            for i, xi in enumerate(PlasmaE.PlasmaParams.xi_bubble[0:-1]):
                myprog.step = i+1
                # ======================================
                # Update positions
                # ======================================
                PlasmaE.x_coords[i+1, :] = PlasmaE.x_coords[i, :] + PlasmaE.bx_coords[i, :] * _spc.speed_of_light * dt
                PlasmaE.y_coords[i+1, :] = PlasmaE.y_coords[i, :] + PlasmaE.by_coords[i, :] * _spc.speed_of_light * dt

                # ======================================
                # Get ion shape
                # ======================================
                self.PlasmaIons.add_ion_ellipse(PlasmaE.x_coords[i], PlasmaE.y_coords[i])
                for j, (x, y, bx, by) in enumerate(zip(PlasmaE.x_coords[i, :], PlasmaE.y_coords[i, :], PlasmaE.bx_coords[i, :], PlasmaE.by_coords[i, :])):
                    # ======================================
                    # Get drive fields at particles
                    # ======================================
                    E = self.Drive.E_fields(x, y, xi) * self.Drive.gamma

                    # ======================================
                    # Get ion fields at particles
                    # ======================================
                    acc = _a(x, y, bx, by, E[0], E[1])

                    # ======================================
                    # Update velocities
                    # ======================================
                    PlasmaE.bx_coords[i+1, j] = PlasmaE.bx_coords[i, j] + acc[0] * dt
                    PlasmaE.by_coords[i+1, j] = PlasmaE.by_coords[i, j] + acc[1] * dt

        self.PlasmaIons.add_ion_ellipse(PlasmaE.x_coords[-1], PlasmaE.y_coords[-1])

        # ======================================
        # Record completion timestamp
        # ======================================
        self._timestamp         = _time.localtime()

        self.PlasmaE._set_timestamp(self._timestamp)
        self.PlasmaIons._set_timestamp(self._timestamp)
        self.Drive._set_timestamp(self._timestamp)

    @property
    def PlasmaE(self):
        """
        The plasma :class:`blowout.electrons.PlasmaE` used for the simulation.
        """
        return self._PlasmaE

    @property
    def PlasmaIons(self):
        """
        The plasma :class:`blowout.ions.PlasmaIons` used for the simulation.
        """
        return self._PlasmaIons

    @property
    def Drive(self):
        """
        The drive :class:`blowout.drive.Drive` used for the simulation.
        """
        return self._Drive

    def write(self, filename=None):
        self.PlasmaE.PlasmaParams.write(filename=filename)
        self.PlasmaIons.write(filename=filename)
        self.PlasmaE.write(filename=filename)
        self.Drive.write(filename=filename)
