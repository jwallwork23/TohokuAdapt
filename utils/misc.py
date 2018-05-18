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
        peak_i, peak = fv.max()
    dgCoords = Function(FunctionSpace(mesh, op.space2, op.degree2)).interpolate(mesh.coordinates[0])
    with dgCoords.dat.vec_ro as dv:         # TODO: I think an error may be occurring here-ish in parallel
        val = np.abs(dv.getValue(peak_i))

    return peak, val


def indicator(mesh, mirror=False, op=Options()):       # TODO: Consider radial indicators, rather than boxes
    """
    :arg mesh: mesh to use.
    :param mirror: consider 'mirror image' indicator region.
    :param op: options parameter class.
    :return: ('Smoothened') indicator function for region A = [x1, x2] x [y1, y1]
    """
    smooth = True if op.mode == 'tohoku' else False
    P0 = FunctionSpace(mesh, "DG", 0)

    # Define extent of region A
    xy = op.xy2 if mirror else op.xy
    iA = Function(P0, name="Region of interest")
    if smooth:
        xd = (xy[1] - xy[0]) / 2
        yd = (xy[3] - xy[2]) / 2
        iA.interpolate(Expression('(x[0] > %f - eps) && (x[0] < %f + eps) && (x[1] > %f - eps) && (x[1] < %f) + eps ? ' \
              'exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) : 0.' \
              % (xy[0], xy[1], xy[2], xy[3], xy[0] + xd, xd, xy[2] + yd, yd), eps=1e-10))
    else:
        iA.interpolate(Expression(
            '(x[0] > %f - eps) && (x[0] < %f + eps) && (x[1] > %f - eps) && (x[1] < %f + eps) ? 1. : 0.' % (
            xy[0], xy[1], xy[2], xy[3]),
            eps=1e-10))

    return iA
