from firedrake import *

import numpy as np
import cmath

from . import interpolation
from . import options


def explicitErrorEstimator(u_, u, eta_, eta, lu, le, b, dt, hk):
    """
    Estimate error locally using an a posteriori error indicator.
    
    :param u_: fluid velocity at previous timestep.
    :param u: fluid velocity at current timestep.
    :param eta_: free surface displacement at previous timestep.
    :param eta: free surface displacement at current timestep.
    :param lu: adjoint velocity at current timestep.
    :param le: adjoint free surface at current timestep.
    :param b: bathymetry field.
    :param dt: timestep used stepping from u_ to u (and eta_ to eta).
    :param hk: cell size of current mesh.
    :return: field of local error indicators.
    """

    # Get useful objects related to mesh
    mesh = u.function_space().mesh()
    v = TestFunction(FunctionSpace(mesh, "DG", 0))      # DG test functions to get cell-wise norms

    # Compute element residual
    rho = assemble(v * pow(hk, 2) * (dot(u_ - u - dt * 9.81 * grad(0.5 * (eta + eta_)),
                                         u_ - u - dt * 9.81 * grad(0.5 * (eta + eta_)))
                                     + (eta_ - eta - dt * div(b * 0.5 * (u + u_)))
                                     * (eta_ - eta - dt * div(b * 0.5 * (u + u_)))) / CellVolume(mesh) * dx)

    # Compute and add boundary residual term
    # TODO: this only currently integrates over domain the boundary, NOT cell boundaries
    # TODO: also, need multiply by normed bdy size
    rho_bdy = assemble(v * dt * b * dot(0.5 * (u + u_), FacetNormal(mesh)) / CellVolume(mesh) * ds)
    lambdaNorm = assemble(v * sqrt((dot(lu, lu) + le * le)) * dx)
    rho = assemble((sqrt(rho) + sqrt(rho_bdy)) * lambdaNorm)

    return Function(FunctionSpace(mesh, "CG", 1)).interpolate(rho)


def DWR(residual, adjoint, v):
    """
    :param residual: approximation of residual for primal equations. 
    :param adjoint: approximate solution of adjoint equations.
    :param v: P0 test function over the same function space.
    :return: dual weighted residual.
    """
    m = len(residual.function_space().dof_count)
    n = len(adjoint.function_space().dof_count)
    assert(m == n)
    if n == 1:
        return assemble(v * inner(residual, adjoint) * dx)
    else:
        return assemble(v * sum([inner(residual.split()[k], adjoint.split()[k]) for k in range(n)]) * dx)


def basicErrorEstimator(primal, dual, v):
    """
    :param primal: approximate solution of primal equations. 
    :param dual: approximate solution of dual equations.
    :param v: P0 test function over the same function space.
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


def FourierSeriesSW(eta0, t, b, trunc=10):
    """
    :param eta0: initial free surface displacement field.
    :param t: current time.
    :param b: (flat) bathymetry.
    :param trunc: term at which to truncate Fourier expansion.
    :return: Fourier series solution for free surface displacement in the shallow water equations.
    """
    V = eta0.function_space()
    eta = Function(V)
    gbSqrt = np.sqrt(9.81 * b)
    pi22 = pow(2 * np.pi, 2)
    for k in range(-trunc, trunc + 1):
        for l in range(-trunc, trunc + 1):
            omega = gbSqrt * np.sqrt(k**2 + l**2)
            cosTerm = Function(V).interpolate(Expression("cos(%.f * x[0] + %.f * x[1])" % (k, l)))
            sinTerm = Function(V).interpolate(Expression("sin(%.f * x[0] + %.f * x[1])" % (k, l)))
            transform = (assemble(cosTerm * eta0 * dx) - cmath.sqrt(-1) * assemble(sinTerm * eta0 * dx)) / pi22
            eta.dat.data[:] += \
                ((cosTerm.dat.data + cmath.sqrt(-1) * sinTerm.dat.data) * omega * transform * cos(omega * t)).real
    return eta

if __name__ == '__main__':
    mode = input("Enter error type to compute (default fixedMesh): ") or 'fixedMesh'
    assert mode in ('fixedMesh', 'adjointBased', 'simpleAdapt')
    lx = 2 * np.pi
    n = 128
    fineMesh = SquareMesh(n, n, lx, lx)
    if input('Hit anything except enter if Fourier expansion not yet computed. '):
        x, y = SpatialCoordinate(fineMesh)
        eta0 = Function(FunctionSpace(fineMesh, "CG", 2)).interpolate(
            1e-3 * exp(- (pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        outfile = File("plots/testSuite/analytic_SW.pvd")
        print("Generating Fourier series solution to linear shallow water equations...")
        for i, t in zip(range(41), np.linspace(0., 2., 41)):
            print("t = %.2f" % t)
            eta = error.FourierSeriesSW(eta0, t, 0.1, trunc=10)
            eta.rename("Fourier series free surface")
            outfile.write(eta, time=t)
            with DumbCheckpoint("plots/testSuite/hdf5/analytic_SW" + options.indexString(i), mode=FILE_CREATE) as chk:
                chk.store(eta)
                chk.close()

    index = 0
    eta = Function(FunctionSpace(fineMesh, "CG", 2), name="Analytic free surface")
    if mode == 'fixedMesh':
        elev_2d = Function(FunctionSpace(SquareMesh(64, 64, lx, lx), "CG", 2), name="elev_2d")
    elif mode == 'simpleAdapt':
        raise NotImplementedError('Need save mesh to load data.')
    elif mode == 'adjointBased':
        raise NotImplementedError('Need save mesh to load data.')
    # TODO: save meshes to compute other error norms

    for index, t in zip(range(41), np.linspace(0., 2., 41)):
        indexStr = options.indexString(index)
        with DumbCheckpoint("plots/testSuite/hdf5/analytic_SW" + indexStr, mode=FILE_READ) as exact:
            exact.load(eta, name="Fourier series free surface")
            exact.close()
        # TODO: save hdf5 to load
        approxn = interpolation.interp(fineMesh, elev_2d)[0]
        print('t = %5.2fs, relative error = %8.6f' % (t, errornorm(approxn, eta) / norm(eta)))