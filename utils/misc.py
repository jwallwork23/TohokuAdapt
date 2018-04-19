from firedrake import *

import numpy as np

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
    # peak_i, peak = getMax(interp(isoP2(mesh_H), f).dat.data)
    peak_i, peak = getMax(f.dat.data)
    dgCoords = Function(VectorFunctionSpace(mesh, op.space2, op.degree2)).interpolate(mesh.coordinates)

    return peak, np.abs(dgCoords.dat.data[peak_i][0])
