from firedrake import *

import numpy as np
import matplotlib.pyplot as plt

import utils.forms as form
import utils.options as opt

Js = []     # Container for objective functional values
diff = 1    # Initialise 'difference of differences'
tol = 5e-4  # Threshold for convergence
i = 0       # Step counter

while diff > tol:

    # Define Mesh and FunctionSpace
    n = pow(2, i)
    mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 2)

    # Specify physical and solver parameters
    op = opt.Options(dt=0.04,
                     Tend=2.4)
    dt = op.dt
    Dt = Constant(dt)
    T = op.Tend
    w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))

    # Apply initial condition and define Functions
    ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
    phi = ic.copy(deepcopy=True)
    phi.rename('Concentration')
    phi_next = Function(V, name='Concentration next')

    # Initialise counters
    t = 0.

    # Define variational problem
    psi = TestFunction(V)
    F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=1e-3)
    bc = DirichletBC(V, 0., "on_boundary")

    iA = form.indicator(V)
    J_trap = assemble(phi * iA * dx)

    print('\nStarting run for resolution %d ' % n)
    finished = False
    while t < T:
        # Solve problem at current timestep
        solve(F == 0, phi_next, bc)

        # Update solution at previous timestep
        phi.assign(phi_next)

        # Estimate OF using trapezium rule
        if t > T:
            finished = True
        step = assemble(phi * iA * dx)
        if finished:
            J_trap += step
        else:
            J_trap += 2 * step
        t += dt
    Js.append(J_trap * dt)
    print("Trapezium rule estimate of J = %.4f" % Js[-1])  # TODO: how to calculate this more elegantly?
    if i > 1:
        diff = np.abs(Js[-2] - Js[-3]) - np.abs(Js[-1] - Js[-2])
        print("Difference in difference : %.4f" % diff)
    i += 1

plt.gcf()
plt.xlabel(r'Mesh resolution')
plt.ylabel(r'Objective functional')
plt.plot([pow(2, i) for i in range(len(Js))], Js)
plt.show()
plt.savefig("outdata/bootstrapping.pdf")
