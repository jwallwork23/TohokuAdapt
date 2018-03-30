from thetis import *

import numpy as np

from .interpolation import mixedPairInterp


__all__ = ["explicitErrorEstimator", "DWR", "fluxJumpError", "basicErrorEstimator", "totalVariation"]


def explicitErrorEstimator(q, residual, b, v, maxBathy=False):
    """
    Estimate error locally using an a posteriori error indicator.
    
    :arg q: primal approximation at current timestep.
    :arg residual: approximation of residual for primal equations.
    :arg b: bathymetry profile.
    :arg v: P0 test function over the same function space.
    :param maxBathy: apply bound on bathymetry.
    :return: field of local error indicators.
    """
    V = residual.function_space()
    m = len(V.dof_count)
    mesh = V.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    b0 = Constant(max(b.dat.data)) if maxBathy else Function(V.sub(1)).interpolate(b)

    # Compute element residual term
    resTerm = assemble(v * h * h * inner(residual, residual) * dx) if m == 1 else \
        assemble(v * h * h * sum([inner(residual.split()[k], residual.split()[k]) for k in range(m)]) * dx)

    # Compute boundary residual term on fine mesh (if necessary)
    if q.function_space().mesh() != mesh:
        q = mixedPairInterp(mesh, V, q)[0]
        u, eta = q.split()

    j0 = assemble(dot(v * grad(u[0]), n) * ds)
    j1 = assemble(dot(v * grad(u[1]), n) * ds)
    j2 = assemble(jump(v * b0 * u, n=n) * dS)
    # jumpTerm = assemble(v * h * j2 * j2 * dx)

    # j0 = assemble(jump(v * grad(u[0]), n=n) * dS)
    # j1 = assemble(jump(v * grad(u[1]), n=n) * dS)
    # j2 = assemble(jump(v * grad(eta), n=n) * dS)
    jumpTerm = assemble(v * h * (j0 * j0 + j1 * j1 + j2 * j2) * dx)

    return assemble(sqrt(resTerm + jumpTerm))


def fluxJumpError(q, v):
    """
    Estimate error locally by flux jump.

    :arg q: primal approximation at current timestep.
    :arg v: P0 test function over the same function space.
    :return: field of local error indicators.
    """
    V = q.function_space()
    mesh = V.mesh()
    h = CellSize(mesh)
    n = FacetNormal(mesh)
    uh, etah = q.split()
    j0 = assemble(jump(v * grad(uh[0]), n=n) * dS)
    j1 = assemble(jump(v * grad(uh[1]), n=n) * dS)
    j2 = assemble(jump(v * grad(etah), n=n) * dS)

    return assemble(v * h * (j0 * j0 + j1 * j1 + j2 * j2) * dx)


def DWR(residual, adjoint, v):
    """
    :arg residual: approximation of residual for primal equations. 
    :arg adjoint: approximate solution of adjoint equations.
    :arg v: P0 test function over the same function space.
    :return: dual weighted residual.
    """
    n = len(adjoint.function_space().dof_count)
    assert(len(residual.function_space().dof_count) == n)
    if n == 1:
        return assemble(v * inner(residual, adjoint) * dx)
    else:
        return assemble(v * sum([inner(residual.split()[k], adjoint.split()[k]) for k in range(n)]) * dx)


def basicErrorEstimator(primal, dual, v):
    """
    :arg primal: approximate solution of primal equations. 
    :arg dual: approximate solution of dual equations.
    :arg v: P0 test function over the same function space.
    :return: error estimate as in DL16.
    """
    m = len(primal.function_space().dof_count)
    n = len(dual.function_space().dof_count)
    assert(m == n)
    if n == 1:
        return assemble(v * inner(primal, dual) * dx)
    else:
        return assemble(v * sum([inner(primal.split()[k], dual.split()[k]) for k in range(n)]) * dx)


def totalVariation(data):
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
