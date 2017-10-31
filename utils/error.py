from firedrake import *

import numpy as np

from . import options


class OutOfRangeError(ValueError):
    pass


def explicitErrorEstimator(W, u_, u, eta_, eta, lu, le, b, dt, op=options.Options()):
    """
    Estimate error locally using an a posteriori error indicator.
    
    :param W: mixed function space upon which variables are defined.
    :param u_: fluid velocity at previous timestep.
    :param u: fluid velocity at current timestep.
    :param eta_: free surface displacement at previous timestep.
    :param eta: free surface displacement at current timestep.
    :param lu: adjoint velocity at current timestep.
    :param le: adjoint free surface at current timestep.
    :param b: bathymetry field.
    :param dt: timestep used stepping from u_ to u (and eta_ to eta).
    :param op: Options class object providing min/max cell size values.
    :return: field of local error indicators.
    """

    # Get useful objects related to mesh
    mesh = W.mesh()
    v = TestFunction(FunctionSpace(mesh, "DG", 0))  # DG test functions to get cell-wise norms
    hk = Function(W.sub(1)).interpolate(CellSize(mesh))

    # Compute element residual
    rk_01 = Function(W.sub(0)).interpolate(u_ - u - dt * op.g * grad(0.5 * (eta + eta_)))
    rk_2 = Function(W.sub(1)).interpolate(eta_ - eta - dt * div(b * 0.5 * (u + u_)))
    rho = assemble(v * hk * sqrt(dot(rk_01, rk_01) + rk_2 * rk_2) / CellVolume(mesh) * dx)

    # Compute and add boundary residual term
    # TODO: this only currently integrates over domain the boundary, NOT cell boundaries
    # TODO: also, need multiply by normed bdy size
    rho += assemble(v * dt * b * dot(0.5 * (u + u_), FacetNormal(mesh)) * ds)
    lambdaNorm = assemble(v * sqrt((dot(lu, lu) + le * le)) * dx)
    rho *= lambdaNorm
    rho.rename("Local error indicators")

    return rho


def basicErrorEstimator(u, eta, lu, le):
    """
    Consider significant regions as those where the 'dot product' between forward and adjoint variables take significant
    values in modulus, as per Davis & LeVeque 2016.
    
    :param u: fluid velocity at current timestep.
    :param eta: free surface displacement at current timestep.
    :param lu: adjoint fluid velocity at current timestep.
    :param le: adjoint free surface displacement at current timestep.
    :return: field of local error indicators taken as product over fields
    """
    rho = Function(eta.ufl_function_space()).assign(eta * le)
    rho.dat.data[:] += u.dat.data[:, 0] * lu.dat.data[:, 0] + u.dat.data[:, 1] * lu.dat.data[:, 1]
    rho.rename("Local error indicators")

    return rho


def totalVariation(data):
    """
    :param data: (one-dimensional) timeseries record.
    :return: total variation thereof.
    """
    TV = 0
    iStart = 0
    for i in range(1, len(data)):
        if i == 1:
            sign = (data[i] - data[i - 1]) / np.abs(data[i] - data[i - 1])
        else:
            sign_ = sign
            sign = (data[i] - data[i - 1]) / np.abs(data[i] - data[i - 1])
            if (sign != sign_) | (i == len(data) - 1):
                TV += np.abs(data[i] - data[iStart])
                iStart = i
    return TV
