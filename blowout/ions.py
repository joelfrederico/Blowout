from .support import _timestamp2filename
from .support import Timestamp as _Timestamp
from .support import _write_arrays
from .support import _write_scalars
from .support import _write_data
import numpy as _np
import scisalt as _ss
import skimage.measure as _skmeas
import skimage.morphology as _skmorph
import skimage.segmentation as _skseg
import skimage.transform as _sktrans
import skimage.draw as _skdraw
import time as _time
import h5py as _h5
import pkg_resources as _pkg_resources
import logging as _logging
import ipdb as pdb

_version = _pkg_resources.get_distribution('blowout').version
_logger = _logging.getLogger(__name__)

import ipdb


class PlasmaIons(_Timestamp):
    def __init__(self, PlasmaParams):
        super().__init__()
        self._PlasmaParams = PlasmaParams
        self._step_ind     = 0

        num_steps = PlasmaParams.num_steps

        self._img            = _np.empty(num_steps, dtype=object)
        self._xind           = _np.empty(num_steps)
        self._yind           = _np.empty(num_steps)
        self._closed_ellipse = _np.empty(num_steps, dtype=object)
        self._prop           = _np.empty(num_steps, dtype=object)
        self._bounds         = _np.empty(num_steps, dtype=object)
        self._results        = _np.empty(num_steps, dtype=object)

    def _save_results(self):
        results = self._results
        # ======================================
        # Get names in results
        # ======================================
        names = self._results[0].dtype.names

        # ======================================
        # Create dict for flattened array
        # ======================================
        dict_layout = {'names': [], 'formats': []}
        for name in names:
            dict_layout['names'].append(name)
            maxlen = 0
            for result in results:
                res_name_len = len(result[name])
                if res_name_len > maxlen:
                    maxlen = res_name_len

            if res_name_len > 1:
                dict_layout['formats'].append(object)
            else:
                dict_layout['formats'].append(results[0][name].dtype)

        results_flat = _np.zeros(len(results), dtype=dict_layout)
        # ======================================
        # Create flattened array
        # ======================================
        for i, result in enumerate(results):
            for name in names:
                results_flat[name][i] = result[name]
        
        ipdb.set_trace()
        self._results_flat = results_flat

    @property
    def PlasmaParams(self):
        """
        The base plasma parameters.
        """
        return self._PlasmaParams

    def _set_timestamp(self, timestamp):
        self._timestamp = timestamp
        self._PlasmaParams._set_timestamp(timestamp)

    def write(self, filename=None):
        """

        Write all of the particles and plasma parameters to a file.
        """
        filename = _timestamp2filename(self, ftype='ions', filename=filename)

        # ======================================
        # Create flat results
        # ======================================
        self._save_results()

        # ======================================
        # Create new filename
        # ======================================
        with _h5.File(filename, 'w') as f:
            # ipdb.set_trace()
            f.attrs['version'] = _version
            # f.attrs.create(name='version', data=_version)
            
            gdata = f.create_group('data')

            # ======================================
            # Write data
            # ======================================
            dxind           = _write_scalars(group=gdata, name='xind', data=self._xind)          # noqa
            dyind           = _write_scalars(group=gdata, name='yind', data=self._xind)          # noqa
            dimg            = _write_arrays(group=gdata , name='img'            , data=self._img            )  # noqa
            dclosed_ellipse = _write_arrays(group=gdata , name='closed_ellipse' , data=self._closed_ellipse )  # noqa
            dbounds         = _write_arrays(group=gdata , name='bounds'         , data=self._bounds         )  # noqa

            gresults_flat = gdata.create_group('results_flat')
            names = self._results_flat.dtype.names
            for name in names:
                _write_data(gresults_flat, name, self._results_flat[name])

            gmeta = f.create_group('metadata')  # noqa
            # gmeta.attrs.create(name='num_parts' , data=self.num_parts )

    def draw_ellipse(self, i=0):
        data = self._results_flat
        j = _np.argmax(data['count_density'][i])
        x, y = _skdraw.ellipse_perimeter(
            cy          = data['yc'][i][j].astype('int'),
            cx          = data['xc'][i][j].astype('int'),
            yradius     = data['a'][i][j].astype('int'),
            xradius     = data['b'][i][j].astype('int'),
            orientation = data['orientation'][i][j]
            )
        
        bounds_int = self._bounds[i].astype('int')
        
        bounds_int[x, y] = 2
    
        _ss.matplotlib.Imshow_Slider(bounds_int)
        # plt.show()

    def add_ion_ellipse(self, x, y, step_ind=None):
        t = _time.perf_counter()
        _logger.debug('Finding ellipse...')
        # print('Finding ellipse...')
        if step_ind is None:
            step_ind = self._step_ind
            self._step_ind = step_ind + 1
            # print('Step: {}'.format(step_ind))

        # ======================================
        # Histogram particles
        # ======================================
        # ind = _np.abs(x) < 3
        # x = x[ind]
        # y = y[ind]
        img, extent = _ss.matplotlib.hist2d(x, y, bins=200, plot=False)
        self._img[step_ind] = img
        
        # ======================================
        # Find index of center
        # ======================================
        xind, yind = _imgcenter(img, extent)
        self._xind[step_ind] = xind
        self._yind[step_ind] = yind
        
        # ======================================
        # Find ellipse corresponding to center
        # region
        # ======================================
        labels = _skmeas.label(img)
        cent_label = labels[xind, yind]
        ellipse = (labels == cent_label)
        
        # ======================================
        # Find ellipse closing (smooth gaps)
        # ======================================
        selem = _skmorph.square(3)
        closed_ellipse = _skmorph.binary_closing(ellipse, selem=selem)
        self._closed_ellipse[step_ind] = ellipse
        
        # ======================================
        # Get properties of the ellipse
        # ======================================
        props = _skmeas.regionprops(closed_ellipse)
        prop = props[0]
        self._prop[step_ind] = prop
        
        # ======================================
        # Find ellipse edges
        # ======================================
        bounds = _skseg.find_boundaries(closed_ellipse.astype('int'), mode='subpixel')
        self._bounds[step_ind] = bounds
        
        # ======================================
        # Find ellipse
        # ======================================
        xmean, ymean = _np.array(prop.centroid)*2
        results = _ss.scipy.hough_ellipse(bounds, xmean=xmean, ymean=ymean, threshold=5)
        # results = _ss.scipy.hough_ellipse(bounds, threshold=1)

        _logger.debug('Found: {} s'.format(_time.perf_counter()-t))
        # print('Found: {} s'.format(_time.perf_counter()-t))

        self._results[step_ind] = results
    
        return results


def _imgcenter(img, extent):
    # x0 = (extent[1]+extent[0])/2
    # y0 = (extent[3]+extent[2])/2
    lx = extent[1]-extent[0]
    ly = extent[3]-extent[2]

    mx = img.shape[0] / lx
    my = img.shape[0] / ly
    bx = -mx * extent[0]
    by = -my * extent[2]

    return _np.array(_np.round((bx, by)), dtype=int)
