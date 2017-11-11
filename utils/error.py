from firedrake import *

import numpy as np
import cmath

from . import interpolation
from . import storage


class OutOfRangeError(ValueError):
    pass


def explicitErrorEstimator(W, u_, u, eta_, eta, lu, le, b, dt):
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
    :return: field of local error indicators.
    """

    # Get useful objects related to mesh
    mesh = W.mesh()
    v = TestFunction(FunctionSpace(mesh, "DG", 0))  # DG test functions to get cell-wise norms
    hk = Function(W.sub(1)).interpolate(CellSize(mesh))

    # Compute element residual
    rk_01 = Function(W.sub(0)).interpolate(u_ - u - dt * 9.81 * grad(0.5 * (eta + eta_)))
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


def basicErrorEstimator(u, lu, eta, le):
    """
    Consider significant regions as those where the 'dot product' between forward and adjoint variables take significant
    values in modulus, as per Davis & LeVeque 2016.
    
    :param u: fluid velocity at current timestep.
    :param lu: adjoint fluid velocity at current timestep.
    :param eta: free surface displacement at current timestep.
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


def analyticSolutionSW(V, b, t, x0=0., y0=0., h=1e-3, trunc=10):
    """
    :param V: FunctionSpace on which to define analytic solution.
    :param b: (constant) water depth.
    :param t: current time.
    :param x0: x-coordinate of initial Gaussian bell centroid.
    :param y0: y-coordinate of initial Gaussian bell centroid.
    :param h: initial free surface height.
    :param trunc: term of Fourier series at which to truncate.
    :return: analytic solution for linear SWEs on a flat bottom model domain.
    """

    # Collect mesh data and establish functions
    xy = V.mesh().coordinates.dat.data
    eta = Function(V)
    try:
        assert((V.ufl_element().family() == 'Lagrange') & (V.ufl_element().degree() == 1))
    except:
        raise NotImplementedError("Only CG1 fields are currently supported.")

    # TODO: implement quadrature properly
    for k in range(-trunc, trunc + 1):
        for l in range(-trunc, trunc + 1):
            kappa2 = k ** 2 + l ** 2
            omega = np.sqrt(kappa2 * 9.81 * b)
            for j, x, y in zip(range(len(xy)), xy[:, 0], xy[:, 1]):
                eta.dat.data[j] += np.sqrt(kappa2) * np.exp(- kappa2 / 4) * \
                                   np.exp(cmath.sqrt(-1) * (k * (x - x0) + l * (y - y0) - omega * t)).real
    eta.dat.data[:] *= h / (4 * pow(np.pi, 3 / 2))
    return eta


def analyticSolution(eta0, t, b, trunc=10):

    V = eta0.function_space()
    eta = Function(V)
    gbSqrt = np.sqrt(9.81 * b)
    pi22 = pow(2 * np.pi, 2)
    for k in range(-trunc, trunc + 1):
        for l in range(-trunc, trunc + 1):
            kappa = np.sqrt(k**2 + l**2)
            omega = kappa * gbSqrt
            cosTerm = Function(V).interpolate(Expression("cos(%.f * x[0] + %.f * x[1])" % (k, l)))
            sinTerm = Function(V).interpolate(Expression("sin(%.f * x[0] + %.f * x[1])" % (k, l)))
            transform = (assemble(cosTerm * eta0 * dx) - cmath.sqrt(-1) * assemble(sinTerm * eta0 * dx)) / pi22
            eta.dat.data[:] += \
                ((cosTerm.dat.data + cmath.sqrt(-1) * sinTerm.dat.data) * kappa * transform * cos(omega * t)).real
    return eta


if __name__ == '__main__':
    lx = 2 * np.pi
    if input("Hit anything except enter to compute analytic solution. "):
        n = 128
        mesh = SquareMesh(n, n, lx, lx)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        outfile = File("plots/analytic/freeSurface.pvd")
        print("Generating analytic solution to linear shallow water equations...")
        for i, t in zip(range(41), np.linspace(0., 2., 41)):
            print("t = %.2f" % t)
            eta = analyticSolutionSW(V, 0.1, t, x0=np.pi, y0=np.pi)
            eta.rename("Analytic free surface")
            outfile.write(eta, time=t)
            with DumbCheckpoint("plots/analytic/hdf5/freeSurface_" + storage.indexString(i), mode=FILE_CREATE) as chk:
                chk.store(eta)
                chk.close()
    else:
        mode = input("Enter error type to compute: ")
        assert mode in ('fixedMesh', 'adjointBased', 'simpleAdapt')
        index = 0
        fineMesh = SquareMesh(128, 128, lx, lx)
        V = FunctionSpace(fineMesh, "CG", 1)
        eta = Function(V, name="Analytic free surface")
        approxn = Function(V, name="Approximation")
        if mode == 'fixedMesh':
            elev_2d = Function(FunctionSpace(SquareMesh(64, 64, lx, lx), "CG", 2), name="elev_2d")
        elif mode == 'simpleAdapt':
            raise NotImplementedError('Need save mesh to load data.')
        elif mode == 'adjointBased':
            raise NotImplementedError('Need save mesh to load data.')
        # TODO: save meshes to compute other error norms
        for index, t in zip(range(41), np.linspace(0., 2., 41)):
            indexStr = storage.indexString(index)
            with DumbCheckpoint("plots/analytic/hdf5/freeSurface_" + indexStr, mode=FILE_READ) as exact:
                exact.load(eta, name="Analytic free surface")
                exact.close()
            with DumbCheckpoint('plots/tests/' + mode + '/hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as approx:
                approx.load(elev_2d, name="elev_2d")
                approx.close()
            approxn.interpolate(interpolation.interp(fineMesh, elev_2d)[0])
            print('t = %5.2fs, relative error = %8.6f' % (t, errornorm(approxn, eta) / norm(eta)))
