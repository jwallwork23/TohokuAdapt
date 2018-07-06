from firedrake import *

import numpy as np

from .options import TohokuOptions, RossbyWaveOptions


__all__ = ["indexString", "peakAndDistance", "bdyRegion"]


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
