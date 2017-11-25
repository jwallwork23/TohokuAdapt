from firedrake import *
from firedrake_adjoint import *

import numpy as np

import utils.forms as form
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor


# Define initial mesh and mesh statistics placeholders
print('************** ADJOINT-BASED ADAPTIVE TSUNAMI SIMULATION **************\n')
# print('ADJOINT-GUIDED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = msh.TohokuDomain(4)
# nEle, nVer = msh.meshStats(mesh)
# N = [nEle, nEle]    # Min/max #Elements
# print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d.' % (nEle, nVer))
# Interpolate bathymetry in DG space

b = Constant(3000.)     # TODO: don't assume flat bathymetry

# Get default parameter values and check CFL criterion
op = opt.Options()
dirName = 'plots/adjointBased/discrete/'
T = op.T
dt = op.dt
Dt = Constant(dt)
cdt = op.hmin / np.sqrt(op.g * max(b.dat.data))
op.checkCFL(b)
ndump = op.ndump
rm = op.rm

# # Initialise counters, constants and function space
# iStart = int(op.Ts / (dt * rm))     # Index corresponding to tStart
# iEnd = int(np.ceil(T / (dt * rm)))  # Index corresponding to tEnd

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
F = form.weakResidualSW(q, q_, qt, b, Dt)

# Define Function to hold residual data
rho = Function(Q, name='Residual')

# Initialise counters and time integrate
t = 0.
cnt = meshn = 0
finished = False
forwardFile = File(dirName + "forward.pvd")
residualFile = File(dirName + "residual.pvd")
forwardFile.write(u, eta, time=t)
while t <= T:
    # Solve problem at current timestep
    solve(F == 0, q)
    q_.assign(q)

    if meshn == 0:

        # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
        if t >= T - dt:
            finished = True
        if t == 0.:
            adj_start_timestep()
        else:
            adj_inc_timestep(time=t, finished=finished)

        # Approximate residual of forward equation and save to HDF5
        Au, Ae = form.strongResidualSW(q, q_, qt, b, Dt)
        rho_u, rho_e = rho.split()
        rho_u.interpolate(Au)
        rho_e.interpolate(Ae)
        rho_u.rename("Residual for velocity")
        rho_e.rename("Residual for elevation")
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