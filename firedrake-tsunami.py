from firedrake import *
from firedrake_adjoint import *

import utils.forms as form
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor


dt_meas = dt
dirName = 'plots/adjointBased/discrete/'

# Define initial mesh
mesh, eta0, b = msh.TohokuDomain(4)

# Get default parameter values and check CFL criterion
op = opt.Options()
op.checkCFL(b)
dt = op.dt
Dt = Constant(dt)
T = op.T
ndump = op.ndump

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
rho_u.rename("Velocity residual")
rho_e.rename("Elevation residual")
dual = Function(Q)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Initialise counters and time integrate
t = 0.
cnt = dumpn = 0
finished = False
forwardFile = File(dirName + "forward.pvd")
residualFile = File(dirName + "residual.pvd")
forwardFile.write(u, eta, time=t)
while t <= T - dt:
    # Solve problem at current timestep
    forwardSolver.solve()
    q_.assign(q)

    # Mark timesteps to be used in adjoint simulation
    if t >= T - dt:
        finished = True
    if t == 0.:
        adj_start_timestep()
    else:
        adj_inc_timestep(time=t, finished=finished)

    if dumpn == 0:
        # Approximate residual of forward equation and save to HDF5
        Au, Ae = form.strongResidualSW(q, q_, b, Dt)
        rho_u.interpolate(Au)
        rho_e.interpolate(Ae)
        with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_CREATE) as chk:
            chk.store(rho_u)
            chk.store(rho_e)
            chk.close()

        # Save to vtu
        forwardFile.write(u, eta, time=t)
        residualFile.write(rho_u, rho_e, time=t)
        print('FORWARD: t = %.2fs, Count: %d ' % (t, cnt))
        dumpn += ndump
        cnt += 1
    dumpn -= 1
    t += dt
cnt -= 1

# Set up adjoint problem
J = form.objectiveFunctionalSW(q, plot=True)
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
        if dumpn == ndump:
            # Load adjoint data
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
            adjointFile.write(dual_u, dual_e, time=t)
            errorFile.write(epsilon, time=t)
            print('ADJOINT: t = %.2fs, Count: %d' % (t, cnt))
            dumpn -= ndump
            cnt -= 1
        dumpn += 1
        t -= dt
        if (t <= 0.) | (cnt == 0):
            break
        save = False
    else:
        save = True

# TODO: extend this to cover all cases
