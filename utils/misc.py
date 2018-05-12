from firedrake import *

import numpy as np

from .adaptivity import isoP2
from .interpolation import interp
from .options import Options


__all__ = ["indexString", "cheatCodes", "getMax", "peakAndDistance"]


def indexString(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def cheatCodes(approach, default='DWR'):
    """
    Enable skipping of sections of code using 'saved' and 'regen'.
    
    :arg approach: user specified input.
    :return: approach to use and keys to skip sections.
    """
    approach = approach or default
    if approach in ('norm', 'fieldBased', 'gradientBased', 'hessianBased', 'fluxJump'):
        getData = False
        getError = False
        useAdjoint = False
        aposteriori = False
    elif approach in ('residual', 'explicit', 'implicit', 'DWR', 'DWE', 'DWP'):
        getData = True
        getError = True
        useAdjoint = approach in ('DWR', 'DWE', 'DWP')
        aposteriori = True
    elif approach in ('saved', 'regen'):
        getError = approach == 'regen'
        approach = input("""Choose error estimator from 'residual', 'explicit', 'implicit', 'DWP', 'DWR' or 'DWE': """)\
                   or 'DWR'
        getData = False
        useAdjoint = approach in ('DWR', 'DWE', 'DWP')
        aposteriori = True
    else:
        approach = 'fixedMesh'
        getData = True
        getError = False
        useAdjoint = False
        aposteriori = False

    return approach, getData, getError, useAdjoint, aposteriori


def getMax(array):
    """
    :param array: 1D array.
    :return: index for maximum and its value. 
    """
    i = 0
    m = 0
    for j in range(len(array)):
        if array[j] > m:
            m = array[j]
            i = j
    return i, m


def peakAndDistance(f, op=Options()):
    mesh = f.function_space().mesh()
    # peak_i, peak = getMax(interp(isoP2(mesh), f).dat.data)
    peak_i, peak = getMax(f.dat.data)
    dgCoords = Function(VectorFunctionSpace(mesh, op.space2, op.degree2)).interpolate(mesh.coordinates)

    return peak, np.abs(dgCoords.dat.data[peak_i][0])


def indicator(V, mirror=False, op=Options()):
    """
    :arg V: Function space to use.
    :param mirror: consider 'mirror image' indicator region.
    :param op: options parameter class.
    :return: ('Smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    """
    smooth = True if op.mode == 'tohoku' else False

    # Define extent of region A
    xy = op.xy2 if mirror else op.xy
    if smooth:
        xd = (xy[1] - xy[0]) / 2
        yd = (xy[3] - xy[2]) / 2
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? ' \
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.' \
              % (xy[0], xy[1], xy[2], xy[3], xy[0] + xd, xd, xy[2] + yd, yd)
    else:
        ind = '(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (xy[0], xy[1], xy[2], xy[3])
    iA = Function(V, name="Region of interest").interpolate(Expression(ind))

    return iA
