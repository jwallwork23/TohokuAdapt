from firedrake import *
from thetis import *

import numpy as np
from time import clock

import utils.forms as form
import utils.mesh as msh
import utils.options as opt


def solverAD(n, op = opt.Options(dt=0.04, Tend=2.4)):

    # Define Mesh and FunctionSpace
    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
    nEle = msh.meshStats(mesh)[0]
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)

    # Specify physical and solver parameters
    Dt = Constant(op.dt)
    w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))

    # Apply initial condition and define Functions
    ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
    phi = ic.copy(deepcopy=True)
    phi.rename('Concentration')
    phi_next = Function(V, name='Concentration next')

    # Define variational problem and OF
    psi = TestFunction(V)
    F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=1e-3)
    bc = DirichletBC(V, 0., "on_boundary")
    iA = form.indicator(V)

    J_trap = assemble(phi * iA * dx)
    t = 0.
    while t < op.Tend:
        # Solve problem at current timestep and update variables
        solve(F == 0, phi_next, bc)
        phi.assign(phi_next)

        # Estimate OF using trapezium rule
        step = assemble(phi * iA * dx)
        if t >= op.Tend:
            J_trap += step
        else:
            J_trap += 2 * step
        t += op.dt

    return J_trap * op.dt, nEle


def solverSW(n, op=opt.Options(dt=0.05, Tstart=0.5, Tend=2.5, family='dg-cg',)):

    # Define Mesh and FunctionSpace
    lx = 2 * np.pi
    mesh = SquareMesh(2*n, 2*n, lx, lx)
    nEle = msh.meshStats(mesh)[0]
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    Dt = Constant(op.dt)
    b = Constant(0.1)

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
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)
    iA = form.indicator(V.sub(1), x1=0., x2=0.5*np.pi, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)

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
        t += op.dt

    return J_trap * op.dt, nEle


# TODO: Rossby Wave test problem


def solverFiredrake(coarseness, op=opt.Options()):

    # Define Mesh and FunctionSpace
    mesh, eta0, b = msh.TohokuDomain(op.coarseness)
    nEle = msh.meshStats(mesh)[0]
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    Dt = Constant(op.dt)

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
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)
    iA = form.indicator(V.sub(1), x1=490e3, x2=640e3, y1=4160e3, y2=4360e3, smooth=True)

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
        t += op.dt

    return J_trap * op.dt, nEle


def solverThetis(coarseness, op=opt.Options()):

    # Get Mesh and initial condition and bathymetry defined thereupon
    mesh, eta0, b = msh.TohokuDomain(op.coarseness)
    nEle = msh.meshStats(mesh)[0]
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Set up solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = False
    options.use_grad_depth_viscosity_term = False
    options.simulation_end_time = op.Tend
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.no_exports = True
    options.log_output = True

    # Define OF
    t = 0.
    iA = form.indicator(V.sub(1), x1=490e3, x2=640e3, y1=4160e3, y2=4360e3, smooth=True)
    J_trap = 0.
    def getJ(elev_2d, t, J_trap):
        step = assemble(elev_2d * iA * dx)
        t += op.dt
        if (t == 0.) | (op.Tend - t <= op.dt):
            J_trap += step
        else:
            J_trap += 2 * step

    # Apply ICs and time integrate
    solver_obj.assign_initial_conditions(elev=eta0)
    solver_obj.iterate(export_func=getJ(elev_2d, t, J_trap))

    return J_trap * op.dt, nEle


def bootstrap(problem='advection-diffusion', maxIter=8, tol=1e-3, slowTol=10.):
    Js = []         # Container for objective functional values
    ts = []         # Timing values
    nEls = []       # Container for element counts
    diff = 1        # Initialise 'difference of differences'
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
            J, nEle = solverFiredrake(5-i)
        elif problem == 'thetis-tsunami':
            J, nEle = solverThetis(5-i)
        else:
            raise ValueError("Problem not recognised.")
        t = clock() - tic
        Js.append(J)
        ts.append(t)
        nEls.append(nEle)
        if problem in ('firedrake-tsunami', 'thetis-tsunami'):
            toPrint = 'coarseness = %2d, ' % (5-i)
        else:
            toPrint = 'n = %3d, ' % n
        toPrint += "nEle = %6d, J = %6.4f, run time : %5.3f, " % (nEle, Js[-1], t)
        if i > 0:
            slowdown = ts[-1] / ts[-2]
            toPrint += "slowdown : %5.3f, " % slowdown
        if i > 1:
            diff = np.abs(np.abs(Js[-2] - Js[-3]) - np.abs(Js[-1] - Js[-2]))
            toPrint += "diff : %6.4f" % diff
        print(toPrint)
        iOpt = i+1  # Get current iteration number

        if diff < tol:
            reason = 'attaining tolerance for convergence.'
            break

        if slowdown > slowTol:
            reason = 'run time decreasing too much.'
            break

        if (problem in ('firedrake-tsunami', 'thetis-tsunami')) & (i == 4):
            reason = 'maximum mesh resolution reached.'
            break

    print("Converged to J = %.4f in %d iterations, due to %s" % (Js[-1], iOpt, reason))
    return pow(2, iOpt), Js, nEls
