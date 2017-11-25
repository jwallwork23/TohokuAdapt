from firedrake import *
from firedrake_adjoint import *

import numpy as np

import utils.forms as form
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor

dt_meas = dt


# Define initial mesh and mesh statistics placeholders
print('************** ADJOINT-BASED ADAPTIVE TSUNAMI SIMULATION **************\n')
# print('ADJOINT-GUIDED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = msh.TohokuDomain(4)
# nEle, nVer = msh.meshStats(mesh)
# N = [nEle, nEle]    # Min/max #Elements
# print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d.' % (nEle, nVer))
# Interpolate bathymetry in DG space

# Get default parameter values and check CFL criterion
op = opt.Options()
dirName = 'plots/adjointBased/discrete/'
T = 150. #op.T
dt = op.dt
Dt = Constant(dt)
cdt = op.hmin / np.sqrt(op.g * max(b.dat.data))
op.checkCFL(b)
ndump = op.ndump
rm = op.rm

# Define variables of problem and apply initial conditions
Q = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = Function(Q)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(eta0)
q = Function(Q)
q.assign(q_)
u, eta = q.split()
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Define variational problem
qt = TestFunction(Q)
forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

# Define Function to hold residual data
rho = Function(Q)
rho_u, rho_e = rho.split()
rho_u.rename("Residual for velocity")
rho_e.rename("Residual for elevation")
dual = Function(Q)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Initialise counters and time integrate
t = 0.
cnt = meshn = 0
finished = False
forwardFile = File(dirName + "forward.pvd")
residualFile = File(dirName + "residual.pvd")
forwardFile.write(u, eta, time=t)
while t <= T:
    # Solve problem at current timestep
    forwardSolver.solve()
    q_.assign(q)

    if meshn == 0:

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
        print('FORWARD: t = %.2fs' % t)
        forwardFile.write(u, eta, time=t)
        residualFile.write(rho_u, rho_e, time=t)
        cnt += 1
        meshn += rm
    meshn -= 1
    t += dt
cnt -= 1

# Set up adjoint problem
J = form.objectiveFunctionalSW(eta)
parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
t = T
save = True
adjointFile = File(dirName + "adjoint.pvd")
errorFile = File(dirName + "errorIndicator.pvd")

# P0 test function to extract elementwise values
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)

# Time integrate (backwards)
for (variable, solution) in compute_adjoint(J):

    # TODO: here ``variable`` comes out as all zeros. How to fix this?

    if save:
        if meshn == rm:

            # Load adjoint data
            dual_u.dat.data[:] = variable.dat.data[0]
            dual_e.dat.data[:] = variable.dat.data[1]

            # Load residual data from HDF5
            with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_READ) as chk:
                rho_u = Function(Q.sub(0), name="Residual for velocity")
                rho_e = Function(Q.sub(1), name="Residual for elevation")
                chk.load(rho_u)
                chk.load(rho_e)
                chk.close()

            # Estimate error using forward residual
            epsilon = assemble(v * (inner(rho_u, dual_u) + rho_e * dual_e) * dx)
            epsilon.rename("Error indicator")

            # Print to screen, save data and increment counters
            print('ADJOINT: t = %.2fs' % t)
            adjointFile.write(dual_u, dual_e, time=t)
            errorFile.write(epsilon, time=t)
            cnt -= 1
            if (t <= 0.) | (cnt == 0):
                break
            meshn = 0
        save = False
        t -= dt
    else:
        save = True
    meshn += 1