from firedrake import *
from thetis import *

import numpy as np
from time import clock

from .adaptivity import isoP2
from .forms import indicator, weakResidualAD, weakResidualSW, adjointAD
from .mesh import meshStats, TohokuDomain
from .options import Options


__all__ = ["solverAD", "solverSW", "solverFiredrake", "bootstrap", "continuousAdjointAD", "continuousAdjointSW"]


def solverAD(n, op=Options(Tend=2.4)):

    # Define Mesh and FunctionSpace
    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
    nEle = meshStats(mesh)[0]
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)

    # Specify physical and solver parameters
    w = Function(VectorFunctionSpace(mesh, "CG", 2)).interpolate(Expression([1, 0]))
    h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    dt = min(0.9 * min(h.dat.data), op.Tend / 2)
    Dt = Constant(dt)

    # Apply initial condition and define Functions
    ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
    c_ = ic.copy(deepcopy=True)
    c = Function(V)

    # Define variational problem and OF
    ct = TestFunction(V)
    F = weakResidualAD(c, c_, ct, w, Dt, nu=1e-3)
    iA = indicator(V)

    J_trap = assemble(c_ * iA * dx)
    t = 0.
    while t < op.Tend:
        # Solve problem at current timestep and update variables
        solve(F == 0, c)
        c_.assign(c)

        # Estimate OF using trapezium rule
        step = assemble(c_ * iA * dx)
        if t >= op.Tend:
            J_trap += step
        else:
            J_trap += 2 * step
        t += dt

    return J_trap * dt, nEle


def solverSW(n, op=Options(Tstart=0.5, Tend=2.5, family='dg-cg',)):

    # Define Mesh and FunctionSpace
    lx = 2 * np.pi
    mesh = SquareMesh(2*n, 2*n, lx, lx)
    nEle = meshStats(mesh)[0]
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    b = Constant(0.1)
    h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    dt = min(0.9 * min(h.dat.data) / np.sqrt(op.g * 0.1), op.Tend/2)
    Dt = Constant(dt)

    # Apply initial condition and define Functions
    ic = project(exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V.sub(1))
    q_ = Function(V)
    u_, eta_ = q_.split()
    u_.interpolate(Expression([0, 0]))
    eta_.assign(ic)
    q = Function(V)
    q.assign(q_)
    u, eta = q.split()

    # Define variational problem and OF
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)
    iA = indicator(V.sub(1), x1=0., x2=0.5*np.pi, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)

    started = False
    t = 0.
    while t < op.Tend:

        # Solve problem at current timestep and update variables
        forwardSolver.solve()
        q_.assign(q)

        # Estimate OF using trapezium rule
        step = assemble(eta * iA * dx)
        if (t >= op.Tstart) and not started:
            started = True
            J_trap = step
        elif started:
            J_trap += 2 * step
        if t >= op.Tend:
            J_trap += step
        t += dt

    return J_trap * dt, nEle


def solverFiredrake(nEle, iso=0, op=Options()):

    # Define Mesh and FunctionSpace
    mesh = TohokuDomain(nEle)[0]
    if bool(iso):
        for i in range(int(iso)):
            mesh = isoP2(mesh)
    eta0, b = TohokuDomain(mesh=mesh)[1:]
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    dt = min(0.9 * min(h.dat.data) / np.sqrt(op.g * max(b.dat.data)), op.Tend / 2)
    Dt = Constant(dt)
    print("     Using dt = ", dt)

    # Apply initial condition and define Functions
    q_ = Function(V)
    u_, eta_ = q_.split()
    u_.interpolate(Expression([0, 0]))
    eta_.interpolate(eta0)
    q = Function(V)
    q.assign(q_)
    u, eta = q.split()

    # Define variational problem and OF
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)
    iA = indicator(V.sub(1), x1=490e3, x2=640e3, y1=4160e3, y2=4360e3, smooth=True)

    started = False
    t = 0.
    while t < op.Tend:

        # Solve problem at current timestep and update variables
        forwardSolver.solve()
        q_.assign(q)

        # Estimate OF using trapezium rule
        step = assemble(eta * iA * dx)
        if (t >= op.Tstart) and not started:
            started = True
            J_trap = step
        elif t >= op.Tend:
            J_trap += step
        elif started:
            J_trap += 2 * step
        t += dt

    return J_trap * dt


def bootstrap(problem='advection-diffusion', maxIter=12, tol=1e-3, slowTol=10., op=Options()):
    Js = []         # Container for objective functional values
    ts = []         # Timing values
    nEls = []       # Container for element counts
    diff = 1e20     # Initialise 'difference of differences'
    slowdown = 0
    reason = 'maximum number of iterations being reached.'
    for i in range(maxIter):
        tic = clock()
        if problem == 'advection-diffusion':
            n = pow(2, i)
            J, nEle = solverAD(n)
        elif problem == 'shallow-water':
            n = pow(2, i)
            J, nEle = solverSW(n)
        elif problem == 'rossby-wave':
            raise NotImplementedError
        elif problem == 'firedrake-tsunami':
            nEle = op.meshes[i]
            J = solverFiredrake(nEle)
        elif problem == 'isoP2-firedrake-tsunami':
            nEle = op.meshes[0] * pow(4, i)
            J = solverFiredrake(op.meshes[0], iso=i)
        elif problem == 'thetis-tsunami':
            nEle = op.meshes[i]
            J = solverFiredrake(nEle)
        else:
            raise ValueError("Problem not recognised.")
        t = clock() - tic
        Js.append(J)
        ts.append(t)
        nEls.append(nEle)
        if problem in ('firedrake-tsunami', 'thetis-tsunami', 'isoP2-firedrake-tsunami'):
            toPrint = 'i = %d, ' % i
        else:
            toPrint = 'n = %3d, ' % n
        toPrint += "nEle = %6d, J = %6.4e, run time : %6.3f, " % (nEle, Js[-1], t)
        if i > 0:
            slowdown = ts[-1] / ts[-2]
            toPrint += "slowdown : %5.3f, " % slowdown
            diff = np.abs(Js[-1] - Js[-2])
            toPrint += "diff : %6.4e" % diff
        print(toPrint)
        iOpt = i    # Get current iteration number

        if diff < tol:
            reason = 'attaining tolerance for convergence.'
            break

        if slowdown > slowTol:
            reason = 'run time becoming too high.'
            break

        if (problem in ('firedrake-tsunami', 'thetis-tsunami')) & (i == 11):
            reason = 'maximum mesh resolution reached.'
            break

    print("Converged to J = %6.4e in %d iterations, due to %s" % (Js[-1], iOpt, reason))
    if problem in ('firedrake-tsunami', 'thetis-tsunami'):
        return iOpt, Js, nEls, ts
    else:
        return pow(2, iOpt), Js, nEls, ts


def continuousAdjointSW(n, op=Options(Tstart=0.5, Tend=2.5, family='dg-cg')):

    # Define Mesh and FunctionSpace
    lx = 2 * np.pi
    mesh = SquareMesh(2*n, 2*n, lx, lx)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    b = Constant(0.1)
    h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    dt = min(0.9 * min(h.dat.data) / np.sqrt(op.g * 0.1), op.Tend/2)
    Dt = Constant(dt)

    # Apply initial condition and define Functions
    ic = project(exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V.sub(1))
    q_ = Function(V)
    u_, eta_ = q_.split()
    u_.interpolate(Expression([0, 0]))
    eta_.assign(ic)
    q = Function(V)
    l_ = Function(V)
    lu_, le_ = l_.split()
    lu_.interpolate(Expression([0, 0]))
    le_.interpolate(Expression(0))
    l = Function(V)

    # Define forward problem
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    # Define adjoint problem
    switch = Constant(1.)
    adjointProblem = NonlinearVariationalProblem(weakResidualSW(l, l_, qt, b, Dt, allowNormalFlow=False,
                                                                     adjoint=True, switch=switch), l)
    adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=op.params)

    t = 0.
    cnt = 0
    primalTimer = clock()
    while t < op.Tend - 0.9*dt:
        forwardSolver.solve()
        q_.assign(q)
        t += dt
        cnt += 1
    primalTimer = clock() - primalTimer
    dualTimer = clock()
    while t > 0.1*dt:
        adjointSolver.solve()
        l_.assign(l)
        t -= dt
        cnt -= 1
        if t < op.Tstart:
            switch.assign(0.)
    dualTimer = clock() - dualTimer
    assert (cnt == 0)
    slowdown = dualTimer/primalTimer
    slow = slowdown > 1
    if not slow:
        slowdown = 1./slowdown
    print('Cts case: Adjoint run %.3fx %s than forward run.' % (slowdown, 'slower' if slow else 'faster'))

    return primalTimer, dualTimer, meshStats(mesh)[0]


def continuousAdjointAD(n, op=Options(Tend=2.4)):

    # Define Mesh and FunctionSpace
    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)

    # Specify physical and solver parameters
    w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
    h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    dt = min(0.9 * min(h.dat.data), op.Tend / 2)
    Dt = Constant(dt)

    # Apply initial condition and define Functions
    ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
    c_ = ic.copy(deepcopy=True)
    c = Function(V)
    l_ = Function(V).interpolate(Expression(0))
    l = Function(V)

    # Define forward and adjoint problems
    ct = TestFunction(V)
    F = weakResidualAD(c, c_, ct, w, Dt, nu=1e-3)
    A = adjointAD(l, l_, ct, w, Dt)

    t = 0.
    cnt = 0
    primalTimer = clock()
    while t < op.Tend - 0.5*dt:
        solve(F == 0, c)
        c_.assign(c)
        t += dt
        cnt += 1
    primalTimer = clock() - primalTimer
    dualTimer = clock()
    while t > 0.5*dt:
        solve(A == 0, l)
        l_.assign(l)
        t -= dt
        cnt -= 1
    dualTimer = clock() - dualTimer
    assert (cnt == 0)
    slowdown = dualTimer / primalTimer
    slow = slowdown > 1
    if not slow:
        slowdown = 1. / slowdown
    print('Cts case: Adjoint run %.3fx %s than forward run.' % (slowdown, 'slower' if slow else 'faster'))

    return primalTimer, dualTimer, meshStats(mesh)[0]
