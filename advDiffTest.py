from firedrake import *
from firedrake_adjoint import *

import utils.forms as form
import utils.storage as stor


dt_meas = dt  # Keep a reference to dt, the time-measure of Firedrake
dirName = "plots/advectionDiffusion/"

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
dt = 0.05
Dt = Constant(dt)
w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))

# Establish bilinear form and set boundary conditions
F = form.weakResidualAD(phi_next, phi, psi, w, Dt)
bc = DirichletBC(V, 0., "on_boundary")

# Initialise counters and time integrate
t = 0.
T = 2.5
cnt = 0
finished = False
forwardFile = File(dirName + "forwardAD.pvd")
residualFile = File(dirName + "residualAD.pvd")
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
    rho.interpolate(form.strongResidualAD(phi_next, phi, w, Dt))
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

# Set up adjoint problem
J = Functional(inner(phi, phi)*dx*dt_meas)          # Establish objective functional
parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
t = T
save = True
adjointFile = File("plots/advectionDiffusion/adjointAD.pvd")
errorFile = File("plots/advectionDiffusion/errorIndicatorAD.pvd")
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
            epsilon.rename("Error indicator")

            # Print to screen, save data and increment counters
            print('ADJOINT: t = %.2fs' % t)
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
