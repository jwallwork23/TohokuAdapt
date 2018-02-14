from thetis import *
from thetis.callback import DiagnosticCallback

import numpy as np
import cmath

from . import interpolation
from . import options


class IntegralCallback(DiagnosticCallback):
    """Base class for callbacks that integrals of a scalar quantity in time and space"""
    variable_names = ['current integral', 'objective value']

    def __init__(self, scalar_callback, solver_obj, **kwargs):
        """
        Creates error comparison check callback object

        :arg scalar_callback: Python function that takes the solver object as an argument and
            returns a scalar quantity of interest
        :arg solver_obj: Thetis solver object
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        super(IntegralCallback, self).__init__(solver_obj, **kwargs)
        self.scalar_callback = scalar_callback
        self.objective_value = 0.5 * scalar_callback() * solver_obj.options.timestep

    def __call__(self):
        dt = self.solver_obj.options.timestep
        value = self.scalar_callback() * dt
        if self.solver_obj.simulation_time > self.solver_obj.options.simulation_end_time - 0.5 * dt:
            value *= 0.5
        self.objective_value += value
        return value, self.objective_value

    def message_str(self, *args):
        line = '{0:s} value {1:11.4e}'.format(self.name, args[1])
        return line


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

    # Compute boundary residual term on fine mesh
    qh = interpolation.mixedPairInterp(mesh, V, q)[0]       # TODO: not needed if changing order
    uh, etah = qh.split()

    j0 = assemble(dot(v * grad(uh[0]), n) * ds)
    j1 = assemble(dot(v * grad(uh[1]), n) * ds)
    j2 = assemble(jump(v * b0 * uh, n=n) * dS)
    # jumpTerm = assemble(v * h * j2 * j2 * dx)

    # j0 = assemble(jump(v * grad(uh[0]), n=n) * dS)
    # j1 = assemble(jump(v * grad(uh[1]), n=n) * dS)
    # j2 = assemble(jump(v * grad(etah), n=n) * dS)
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
    m = len(residual.function_space().dof_count)
    n = len(adjoint.function_space().dof_count)
    assert(m == n)
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


def FourierSeriesSW(eta0, t, b, trunc=10):
    """
    :arg eta0: initial free surface displacement field.
    :arg t: current time.
    :arg b: (flat) bathymetry.
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