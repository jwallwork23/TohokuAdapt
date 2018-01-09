from firedrake import *

from time import clock
import numpy as np

import utils.forms as form
import utils.mesh as msh
import utils.options as opt


def continuousAdjointSW(n, op=opt.Options(Tstart=0.5, Tend=2.5, family='dg-cg')):

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
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    # Define adjoint problem
    adjointProblem = NonlinearVariationalProblem(form.weakResidualSW(l, l_, qt, b, Dt, allowNormalFlow=False,
                                                                     adjoint=True), l)
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
    while t > 0.9*dt:
        adjointSolver.solve()
        l_.assign(l)
        t -= dt
        cnt -= 1
    dualTimer = clock() - dualTimer
    assert (cnt == 0)
    slowdown = dualTimer/primalTimer
    slow = slowdown > 1
    if not slow:
        slowdown = 1./slowdown
    print('Continuous case: Adjoint run %.3fx %s than forward run.' % (slowdown, 'slower' if slow else 'faster'))

    # adj_html("outdata/visualisations/forward.html", "forward")
    # adj_html("outdata/visualisations/adjoint.html", "adjoint")

    return primalTimer, dualTimer, msh.meshStats(mesh)[0]

n = pow(2, int(input('Power of 2 for mesh resolution: ') or 1.))
c_t1, c_t2, nEle = continuousAdjointSW(n)
print('Using a mesh with %d elements' % nEle)


from firedrake_adjoint import *
dt_meas = dt  # Time measure


def discreteAdjointSW(n, op=opt.Options(Tstart=0.5, Tend=2.5, family='dg-cg')):

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
    q.assign(q_)
    dual = Function(V)

    # Define variational problem and OF
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)
    J = form.objectiveFunctionalSW(q, Tstart=op.Tstart, x1=0., x2=np.pi/2, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)

    t = 0.
    cnt = 0
    primalTimer = clock()
    while t < op.Tend:
        forwardSolver.solve()
        q_.assign(q)
        if t == 0.:
            adj_start_timestep()
        elif t >= op.Tend:
            adj_inc_timestep(time=t, finished=True)
        else:
            adj_inc_timestep(time=t, finished=False)
        t += dt
        cnt += 1
    cnt -= 1
    primalTimer = clock() - primalTimer
    parameters["adjoint"]["stop_annotating"] = True  # Stop registering equations
    store = True
    dualTimer = clock()
    for (variable, solution) in compute_adjoint(J):
        if store:
            dual.assign(variable, annotate=False)
            cnt -= 1
            store = False
        else:
            store = True
        if cnt == 0:
            break
    dualTimer = clock() - dualTimer
    assert (cnt == 0)
    slowdown = dualTimer/primalTimer
    slow = slowdown > 1
    if not slow:
        slowdown = 1./slowdown
    print('Discrete case: Adjoint run %.3fx %s than forward run.' % (slowdown, 'slower' if slow else 'faster'))

    # adj_html("outdata/visualisations/forward.html", "forward")
    # adj_html("outdata/visualisations/adjoint.html", "adjoint")

    return primalTimer, dualTimer

d_t1, d_t2 = discreteAdjointSW(n)


print("""
            |  Continuous  |  Discrete
-----------------------------------------
Primal run  |   %5.3fs     |   %5.3fs
Dual run    |   %5.3fs     |   %5.3fs
-----------------------------------------
Total       |   %5.3fs     |   %5.3fs""" % (c_t1, d_t1, c_t2, d_t2, c_t1+c_t2, d_t1+d_t2))
