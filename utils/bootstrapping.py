from firedrake import *

import utils.forms as form
import utils.options as opt


# TODO: how to calculate J more elegantly?

def solverAD(n, op = opt.Options(dt=0.04, Tend=2.4)):

    # Define Mesh and FunctionSpace
    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
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

    return J_trap * op.dt


def solverSW(n, op=opt.Options(dt=0.05, Tstart=0.5, Tend=2.5, family='dg-cg',)):

    # Define Mesh and FunctionSpace
    lx = 6
    mesh = SquareMesh(lx*n, lx*n, lx, lx)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    Dt = Constant(op.dt)
    b = Constant(0.1)

    # Apply initial condition and define Functions
    ic = project(1e-3*exp(-(pow(x-3., 2) + pow(y-3., 2))), V.sub(1))
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
    iA = form.indicator(V.sub(1), x1=0., x2=1.5, y1=1.5, y2=4.5, smooth=False)

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

    return J_trap * op.dt
