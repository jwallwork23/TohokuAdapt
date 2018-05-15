from firedrake import *

import numpy as np

from .options import Options


__all__ = ["indexString", "peakAndDistance", "indicator"]


def indexString(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def peakAndDistance(f, op=Options()):
    mesh = f.function_space().mesh()
    with f.dat.vec_ro as fv:
        peak_i, peak = fv.max()[1]
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
