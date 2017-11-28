from firedrake import *
from firedrake_adjoint import *

import numpy as np

import utils.forms as form
import utils.options as opt
import utils.storage as stor


# TODO: integrate into testSuite

dt_meas = dt
dirName = "plots/tests/discreteAdjoint/"

# Define inital mesh and function space
n = 16
lx = 2 * np.pi
mesh = SquareMesh(n, n, lx, lx)
x, y = SpatialCoordinate(mesh)

# Set parameter values
op = opt.Options(dt=0.05, T=2.5)
dt = op.dt
Dt = Constant(dt)
T = op.T
b = Constant(0.1)

# Define variables and apply initial conditions
Q = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = Function(Q)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
q = Function(Q)
q.assign(q_)
u, eta = q.split()
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Define variational problem
qt = TestFunction(Q)
forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

# Define Functions to hold residual and adjoint solution data
rho = Function(Q)
rho_u, rho_e = rho.split()
rho_u.rename("Velocity residual")
rho_e.rename("Elevation residual")
dual = Function(Q)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Initialise counters and time integrate
t = 0.
cnt = 0
finished = False
forwardFile = File(dirName + "forward.pvd")
residualFile = File(dirName + "residual.pvd")
forwardFile.write(u, eta, time=t)
while t <= T:
    # Solve problem at current timestep
    forwardSolver.solve()
    q_.assign(q)

    # Tell dolfin about timesteps, so it can compute functionals including time measures other than dt[FINISH_TIME]
    if t >= T - dt:
        finished = True
    if t == 0.:
        adj_start_timestep()
    else:
        adj_inc_timestep(time=t, finished=finished)

    # Approximate residual of forward equation and save to HDF5
    Au, Ae = form.strongResidualSW(q, q_, b, Dt)
    rho_u.interpolate(Au)
    rho_e.interpolate(Ae)
    with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_CREATE) as chk:
        chk.store(rho_u)
        chk.store(rho_e)
        chk.close()

    # Print to screen, save data and increment counters
    print('FORWARD: t = %.2fs.' % t)
    forwardFile.write(u, eta, time=t)
    residualFile.write(rho_u, rho_e, time=t)
    cnt += 1
    t += dt
cnt -= 1

# Set up adjoint problem
J = form.objectiveFunctionalSW(q, Tstart=0.5, x1=0., x2=np.pi/2, y1=0.5*np.pi, y2=1.5*np.pi, plot=True, smooth=False)
parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
t = T
save = True
adjointFile = File(dirName + "adjoint.pvd")
errorFile = File(dirName + "errorIndicator.pvd")

# P0 test function to extract elementwise values
v = TestFunction(FunctionSpace(mesh, "DG", 0))

# Time integrate (backwards)
for (variable, solution) in compute_adjoint(J):
    if save:
        # Load adjoint data. NOTE the interpolation operator is overloaded
        dual_u.dat.data[:] = variable.dat.data[0]
        dual_e.dat.data[:] = variable.dat.data[1]

        # Load residual data from HDF5
        with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_READ) as chk:
            chk.load(rho_u)
            chk.load(rho_e)
            chk.close()

        # Estimate error using forward residual
        epsilon = assemble(v * inner(rho, dual) * dx)
        epsilon.rename("Error indicator")

        # Print to screen, save data and increment counters
        print('ADJOINT: t = %.2fs.' % t)
        adjointFile.write(dual_u, dual_e, time=t)
        errorFile.write(epsilon, time=t)
        cnt -= 1
        if (t <= 0.) | (cnt == 0):
            break
        t -= dt
        save = False
    else:
        save = True
