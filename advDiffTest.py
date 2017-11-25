from firedrake import *
from firedrake_adjoint import *

import numpy as np

import utils.forms as form


# Define Mesh and FunctionSpace
n = 10
mesh = RectangleMesh(4 * n, n, 4, 1)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
W = VectorFunctionSpace(mesh, "CG", 2)

# Specify initial condition
ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
phi_next = Function(V, name='Concentration next')
v = TestFunction(V)
dual = Function(V, name='Adjoint')
rho = Function(V, name='Residual')

# Specify physical and solver parameters
timestep = Constant(0.05)

# Establish bilinear form and set boundary conditions
w = Function(W, name='Wind field').interpolate(Expression([1, 0]))
F = form.weakResidualAD(phi_next, phi, v, w, timestep)
bc = DirichletBC(V, 0., "on_boundary")

# Initialise counters and time integrate
t = 0.
end = 2.5
finished = False
forwardFile = File("plots/forwardAD.pvd")
residualFile = File("plots/residualAD.pvd")
while (t <= end):
    print('FORWARD: t = %.2fs' % t)

    # Solve problem at current timestep
    solve(F == 0, phi_next, bc)
    phi.assign(phi_next)

    # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
    if t >= end - float(timestep):
        finished = True
    if t == 0.:
        adj_start_timestep()
    else:
        adj_inc_timestep(time=t, finished=finished)
    forwardFile.write(phi, time=t)

    rho.interpolate(form.strongResidualAD(phi_next, phi, w, timestep))
    residualFile.write(rho, time=t)

    # Increment time
    t += float(timestep)

parameters["adjoint"]["stop_annotating"] = True # Stop registering equations

# Establish objective functional
J = Functional(inner(phi, phi)*dx*dt)

# Get solution to adjoint problem       TODO: make this work
t = end
save = True
adjointFile = File("plots/adjointAD.pvd")
for (variable, solution) in compute_adjoint(J):
    try:
        dual.dat.data[:] = variable.dat.data
    except:
        continue
    if save:
        print('ADJOINT: t = %.2fs' % t)
        adjointFile.write(dual, time=t)
        t -= float(timestep)
        save = False
    else:
        save = True
    if t <= 0.:
        break
