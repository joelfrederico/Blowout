import h5py as _h5
import numpy as _np
import logging as _logging
import time as _time
_logger  = _logging.getLogger(__name__)
import ipdb as pdb
import re as _re


def _timestamp2filename(cls, ftype, filename=None):
    # ======================================
    # Get filename from timestamp
    # ======================================
    if filename is not None:
        filename = '{}.{}.h5'.format(filename, ftype)
    else:
        try:
            timestamp = cls.timestamp
        except RuntimeError as err:
            _logger.debug('Handled exception: {}'.format(err))
            timestamp = _time.localtime()

        filename = _time.strftime('%Y.%m.%d.%H%M.%S.{}.h5'.format(ftype), timestamp)

    return filename


class Timestamp(object):
    def __init__(self):
        self._timestamp = None

    def _set_timestamp(self, timestamp):
        self._timestamp = timestamp

    @property
    def timestamp(self):
        if self._timestamp is not None:
            return self._timestamp
        else:
            raise RuntimeError('No timestamp: simulation not completed.')


def _write_arrays(group, name, data, parent=None):
    grefs = group.create_group('_refs_{}'.format(name))
    ref_dtype = _h5.special_dtype(ref=_h5.Reference)
    dname = group.create_dataset(name, (_np.size(data),), dtype=ref_dtype)
    # ======================================
    # Create datasets
    # ======================================
    for i, array in enumerate(data):
        if array.dtype == _np.dtype(object):
            # ======================================
            # If dataset can't be created, nest
            # ======================================
            darray = _write_arrays(grefs, '{}'.format(i), array, parent=name)
        else:
            darray = grefs.create_dataset(name='{}'.format(i), data=array, shape=_np.shape(array), compression="gzip")

        # ======================================
        # Store reference in dataset
        # ======================================
        dname[i] = darray.ref

    # if parent == 'hist':
    #     pdb.set_trace()

    # ======================================
    # Return created dataset
    # ======================================
    return dname


def _read_arrays(group, name):
    refs = group[name]
    arrays = _np.empty(shape=refs.size, dtype=object)
    for i, ref in enumerate(refs):
        arrays[i] = group.file[ref].value

    return arrays


def _write_scalars(group, name, data):
    return group.create_dataset(name=name, data=data, shape=_np.shape(data), compression="gzip")


def _write_data(group, name, data):
    if data.dtype == _np.dtype(object):
        _write_arrays(group, name, data)
    else:
        _write_scalars(group, name, data)


def _read_dict(group, name):
    ret_group = group[name]
    names = ret_group.keys()
    valid_names = list()
    underscore = _re.compile('_')

    dict_layout = {'names': [], 'formats': []}

    for nm in names:
        if not underscore.match(nm):
            valid_names.append(nm)
            dict_layout['names'].append(nm)
            if type(ret_group[nm].value[0]) == _h5.h5r.Reference:
                dict_layout['formats'].append(object)
            else:
                raise NotImplementedError('Haven''t done this...')

    results_flat = _np.zeros(len(ret_group[valid_names[0]]), dtype=dict_layout)

    for nm in valid_names:
        # if nm == 'hist':
        #     pdb.set_trace()
        values = ret_group[nm]
        for i, value in enumerate(values):
            try:
                array = group.file[value].value
                if array.size > 0:
                    if type(array[0]) == _h5.h5r.Reference:
                        out = _np.empty(len(array), dtype=object)
                        for j, val in enumerate(array):
                            out[j] = group.file[val].value
                    else:
                        out = group.file[value].value
                else:
                    out = _np.array([])
                results_flat[nm][i] = out
            except ValueError:
                _logger.debug('There was a ValueError')

    # pdb.set_trace()
    return results_flat
