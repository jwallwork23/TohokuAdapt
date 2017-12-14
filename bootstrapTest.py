from firedrake import *

import numpy as np
import matplotlib.pyplot as plt

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
    lx = 2 * np.pi
    mesh = SquareMesh(n, n, lx, lx)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

    # Specify solver parameters
    Dt = Constant(op.dt)
    b = Constant(0.1)

    # Apply initial condition and define Functions
    ic = project(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V.sub(1))
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
    iA = form.indicator(V.sub(1), x1=0., x2=np.pi/2, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)

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


Js = []         # Container for objective functional values
diff = 1        # Initialise 'difference of differences'
tol = 1e-3      # Threshold for convergence
i = 0           # Step counter
maxIter = 10    # Maximum allowed iterations
advDiff = bool(input("\nHit anything except enter to consider advection-diffusion. "))
filename = "advection-diffusion" if advDiff else "shallow-water"

print("\n ***** " + filename + " *****\n")
for i in range(maxIter):
    n = pow(2, i)
    Js.append(solverAD(n) if advDiff else solverSW(n))
    toPrint = "n = %3d, J = %6.4f, " % (n, Js[-1])
    if i > 1:
        diff = np.abs(np.abs(Js[-2] - Js[-3]) - np.abs(Js[-1] - Js[-2]))
        toPrint += "diff : %6.4f" % diff
    print(toPrint)

    if diff < tol:
        print("Converged to J = %.4f in %d iterations" % (Js[-1], i))
        break
    i += 1

plt.gcf()
plt.xlabel(r'Mesh resolution')
plt.ylabel(r'Objective functional')
plt.semilogx([pow(2, i) for i in range(len(Js))], Js, basex=2)
plt.savefig("outdata/bootstrapping/" + filename + ".pdf", bbox_inches='tight')
plt.show()
