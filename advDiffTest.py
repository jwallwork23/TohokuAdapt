from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.options as opt
import utils.storage as stor

# Establish filenames
dirName = "plots/advectionDiffusion/"
forwardFile = File(dirName + "forwardAD.pvd")
residualFile = File(dirName + "residualAD.pvd")
adjointFile = File(dirName + "adjointAD.pvd")
errorFile = File(dirName + "errorIndicatorAD.pvd")
adaptiveFile = File(dirName + "goalBasedAD.pvd")

# Define Mesh and FunctionSpace
n = 16
mesh = RectangleMesh(4 * n, n, 4, 1)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)

# Specify and apply initial condition
ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
phi_next = Function(V, name='Concentration next')
psi = TestFunction(V)
dual = Function(V, name='Adjoint')
rho = Function(V, name='Residual')

# Specify physical and solver parameters
op = opt.Options(dt=0.025, T=2.5, hmin=5e-2, hmax=1., rm=5)
dt = op.dt
Dt = Constant(dt)
w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
nu = 1e-3   # Diffusivity

# Establish bilinear form and set boundary conditions
F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
bc = DirichletBC(V, 0., "on_boundary")

# Initialise counters and time integrate
t = 0.
T = op.T
cnt = 0
finished = False
print('Starting fixed mesh primal run (forwards in time)')
primalTimer = clock()
while t <= T:
    # Solve problem at current timestep
    solve(F == 0, phi_next, bc)
    phi.assign(phi_next)

    # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
    if t >= T - dt:
        finished = True
    if t == 0.:
        adj_start_timestep()
    else:
        adj_inc_timestep(time=t, finished=finished)

    # Approximate residual of forward equation and save to HDF5
    rho.interpolate(form.strongResidualAD(phi_next, phi, w, Dt, nu=nu))
    with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_CREATE) as chk:
        chk.store(rho)
        chk.close()

    # Print to screen, save data and increment counters
    print('FORWARD: t = %.2fs' % t)
    forwardFile.write(phi, time=t)
    residualFile.write(rho, time=t)
    t += dt
    cnt += 1
cnt -= 1
primalTimer = clock() - primalTimer
print('Primal run complete. Run time: %.2fs' % primalTimer)

# Set up adjoint problem
J = form.objectiveFunctionalAD(phi)
parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
t = T
save = True

# Time integrate (backwards)
print('Starting fixed mesh dual run (backwards in time)')
dualTimer = clock()
for (variable, solution) in compute_adjoint(J):
    try:
        dual.dat.data[:] = variable.dat.data
        if save:
            # Load residual data from HDF5
            with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_READ) as chk:
                rho = Function(V, name='Residual')
                chk.load(rho)
                chk.close()

            # Estimate error using forward residual
            epsilon = assemble(v * rho * dual * dx)
            epsilon.dat.data[:] = np.abs(epsilon.dat.data) * 1e10
            epsilon.rename("Error indicator")

            # Save error indicator data to HDF5
            with DumbCheckpoint(dirName + 'hdf5/error_' + stor.indexString(cnt), mode=FILE_CREATE) as chk:
                chk.store(epsilon)
                chk.close()

            # Print to screen, save data and increment counters
            print('t = %.2fs' % t)
            adjointFile.write(dual, time=t)
            errorFile.write(epsilon, time=t)
            t -= dt
            cnt -= 1
            save = False
        else:
            save = True
        if (t <= 0.) | (cnt == 0):
            break
    except:
        continue
dualTimer = clock() - dualTimer
print('Adjoint run complete. Run time: %.2fs' % dualTimer)

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm

# Reset initial conditions for primal problem and recreate error indicator placeholder
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
epsilon = Function(P0, name="Error indicator")

print('Starting adaptive mesh primal run (forwards in time)')
adaptTimer = clock()
while t <= T:
    # Load error indicator data from HDF5
    with DumbCheckpoint(dirName + 'hdf5/error_' + stor.indexString(cnt), mode=FILE_READ) as chk:
        chk.load(epsilon)   # Defined on a P0 field on the initial mesh
        chk.close()

    # Adapt mesh
    V = TensorFunctionSpace(mesh, "CG", 1)
    H = adap.constructHessian(mesh, V, phi, op=op)

    # TODO: adapt mesh, interpolate
    # TODO: redefine problem

    # Solve problem at current timestep
    solve(F == 0, phi_next, bc)
    phi.assign(phi_next)

    # Print to screen, save data and increment counters
    print('t = %.2fs' % t)
    adaptiveFile.write(phi, time=t)
    t += dt
    cnt += 1
cnt -= 1
adaptTimer = clock() - adaptTimer
print('Primal run complete.')

# Print to screen timing analyses
print("""******** TIMINGS ********
forward run   %5.2fs
adjoint run   %5.2fs
adaptive run  %5.2fs""" % (primalTimer, dualTimer, adaptTimer))
