from firedrake import *

import numpy as np

from .options import TohokuOptions, RossbyWaveOptions


__all__ = ["indexString", "peakAndDistance", "indicator", "bdyRegion"]


def indexString(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def peakAndDistance(f, op=RossbyWaveOptions()):
    mesh = f.function_space().mesh()
    with f.dat.vec_ro as fv:
        peak_i, peak = fv.max()
    dgCoords = Function(FunctionSpace(mesh, 'DG' if op.family == 'dg-dg' else 'CG',  1 if op.family == 'dg-dg' else 2))
    dgCoords.interpolate(mesh.coordinates[0])
    with dgCoords.dat.vec_ro as dv:         # TODO: I think an error may be occurring here-ish in parallel
        val = np.abs(dv.getValue(peak_i))

    return peak, val


def indicator(mesh, xy=None, mirror=False, radii=None, op=TohokuOptions()):       # TODO: Update AD and remove this
    """
    :arg mesh: mesh to use.
    :arg xy: Custom selection for indicator region.
    :param mirror: consider 'mirror image' indicator region.
    :param radii: consider disc of radius `radius`, as opposed to rectangle.
    :param op: options parameter class.
    :return: ('Smoothened') indicator function for region A.
    """

    # Define extent of region A
    if xy is None:
        xy = op.xy2 if mirror else op.xy
    iA = Function(FunctionSpace(mesh, "DG", 1), name="Region of interest")

    if radii is not None:
        if len(np.shape(radii)) == 0:
            expr = Expression("pow(x[0] - x0, 2) + pow(x[1] - y0, 2) < r + eps ? 1 : 0",
                              x0=xy[0], y0=xy[1], r=pow(radii, 2), eps=1e-10)
        elif len(np.shape(radii)) == 1:
            assert len(xy) == len(radii)
            e = "(pow(x[0] - %f, 2) + pow(x[1] - %f, 2) < %f + %f)" % (xy[0][0], xy[0][1], pow(radii[0], 2), 1e-10)
            for i in range(1, len(radii)):
                e += "&& (pow(x[0] - %f, 2) + pow(x[1] - %f, 2) < %f + %f)" \
                     % (xy[i][0], xy[i][1], pow(radii[i], 2), 1e-10)
            expr = Expression(e)
        else:
            raise ValueError("Indicator function radii input not recognised.")
    else:
        if op.mode == 'tohoku':
            xd = (xy[1] - xy[0]) / 2
            yd = (xy[3] - xy[2]) / 2
            expr = Expression("(x[0] > %f - eps) && (x[0] < %f + eps) && (x[1] > %f - eps) && (x[1] < %f) + eps ? "
                              "exp(1. / (pow(x[0] - %f, 2) - pow(%f, 2))) * exp(1. / (pow(x[1] - %f, 2) - pow(%f, 2))) "
                              ": 0." % (xy[0], xy[1], xy[2], xy[3], xy[0] + xd, xd, xy[2] + yd, yd), eps=1e-10)
        else:
            expr = Expression(
                "(x[0] > %f - eps) && (x[0] < %f + eps) && (x[1] > %f - eps) && (x[1] < %f + eps)"
                % (xy[0], xy[1], xy[2], xy[3]), eps=1e-10)
    iA.interpolate(expr)

    return iA


def bdyRegion(mesh, bdyTag, scale, sponge=False):

    bc = DirichletBC(FunctionSpace(mesh, "CG", 1), 0, bdyTag)
    coords = mesh.coordinates.dat.data

    xy  = []
    for i in bc.nodes:
        xy.append(coords[i])

    e = "exp(-(pow(x[0] - %f, 2) + pow(x[1] - %f, 2)) / %f)" % (xy[0][0], xy[0][1], scale)
    for i in range(1, len(xy)):
        e += "+ exp(-(pow(x[0] - %f, 2) + pow(x[1] - %f, 2)) / %f)" % (xy[i][0], xy[i][1], scale)
    # f = "sqrt(pow(x[0] - %f, 2) + pow(x[1] - %f, 2)) / %f)" % (xy[0][0], xy[0][1], scale)
    if sponge:
        expr = Expression(e + " < 1e-3 ? 1e-3 : abs (" + e + ")")   # TODO: Needs redoing
    else:
        expr = Expression(e + " > 1 ? 1 : " + e)

    return expr

