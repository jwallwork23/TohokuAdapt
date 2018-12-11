from firedrake import *

import scipy.interpolate as si
import numpy as np
import h5py

from .options import TohokuOptions, AdvectionOptions


__all__ = ["index_string", "peak_and_distance", "boundary_region", "extract_slice", "extract_gauge_data",
           "gauge_total_variation"]


def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def boundary_region(mesh, bdyTag, scale, sponge=False):

    bc = DirichletBC(FunctionSpace(mesh, "CG", 1), 0, bdyTag)
    coords = mesh.coordinates.dat.data

    xy  = []
    for i in bc.nodes:
        xy.append(coords[i])

    e = "exp(-(pow(x[0] - {x0:f}, 2) + pow(x[1] - {y0:f}, 2)) / {a:f})".format(x0=xy[0][0], y0=xy[0][1], a=scale)
    for i in range(1, len(xy)):
        e += "+ exp(-(pow(x[0] - {x0:f}, 2) + pow(x[1] - {y0:f}, 2)) / {a:f})".format(x0=xy[i][0], y0=xy[i][1], a=scale)
    # f = "sqrt(pow(x[0] - %f, 2) + pow(x[1] - %f, 2)) / %f)" % (xy[0][0], xy[0][1], scale)
    if sponge:
        expr = Expression(e + " < 1e-3 ? 1e-3 : abs (" + e + ")")   # TODO: Needs redoing. Also needs to avoid C-strings
    else:
        expr = Expression(e + " > 1 ? 1 : " + e)

    return expr


def extract_slice(quantities, direction='h', op=AdvectionOptions()):
    if direction == 'h':
        sl = op.h_slice
        label = 'horizontal'
    elif direction == 'v':
        sl = op.v_slice
        label = 'vertical'
    else:
        raise NotImplementedError("Only horizontal and vertical slices are currently implemented.")
    hf = h5py.File(op.directory() + 'diagnostic_' + label + '_slice.hdf5', 'r')
    for x in ["{l:s}_slice{i:d}".format(l=direction, i=i) for i in range(len(sl))]:
        vals = np.array(hf.get(x))
        for i in range(len(vals)):
            tag = '{l:s}_snapshot_{i:d}'.format(l=direction, i=i)
            if not tag in quantities.keys():
                quantities[tag] = []
            quantities[tag].append(vals[i])
    hf.close()


def extract_gauge_data(quantities, op=TohokuOptions()):
    hf = h5py.File(op.directory() + 'diagnostic_timeseries.hdf5', 'r')
    for g in op.gauges:
        if not g in quantities.keys():
            quantities[g] = ()
        quantities[g] += tuple(hf.get(g))
    hf.close()


def extract_spline(gauge):
    measured_file = open('resources/gauges/'+gauge+'data_25mins.txt', 'r')
    x = []
    y = []
    for line in measured_file:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    spline = si.interp1d(x, y, kind=1)
    measured_file.close()
    return spline


def total_variation(data):
    """
    :arg data: (one-dimensional) timeseries record.
    :return: total variation thereof.
    """
    TV = 0
    iStart = 0
    for i in range(len(data)):
        if i == 1:
            sign = (data[i] - data[i-1]) / np.abs(data[i] - data[i-1])
        elif i > 1:
            sign_ = sign
            sign = (data[i] - data[i - 1]) / np.abs(data[i] - data[i - 1])
            if sign != sign_:
                TV += np.abs(data[i-1] - data[iStart])
                iStart = i-1
                if i == len(data)-1:
                    TV += np.abs(data[i] - data[i-1])
            elif i == len(data)-1:
                TV += np.abs(data[i] - data[iStart])
    return TV


def gauge_total_variation(data, gauge="P02"):
    """
    :param data: timeseries to calculate error of.
    :param gauge: gauge considered.
    :return: total variation.
    """
    N = len(data)
    spline = extract_spline(gauge)
    times = np.linspace(0., 25., N)
    errors = [data[i] - spline(times[i]) for i in range(N)]
    return total_variation(errors) / total_variation([spline(times[i]) for i in range(N)])
