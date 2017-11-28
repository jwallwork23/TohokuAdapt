from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor

print('\n*********************** SHALLOW WATER TEST PROBLEM ************************\n')
print('Mesh adaptive solver initially defined on a square mesh')
useAdjoint = bool(input("Hit anything except enter to use adjoint equations to guide adaptive process. "))

# Establish filenames
dirName = "plots/testSuite/"
forwardFile = File(dirName + "forwardSW.pvd")
residualFile = File(dirName + "residualSW.pvd")
adjointFile = File(dirName + "adjointSW.pvd")
errorFile = File(dirName + "errorIndicatorSW.pvd")
adaptiveFile = File(dirName + "goalBasedSW.pvd") if useAdjoint else File(dirName + "simpleAdaptSW.pvd")

# Specify physical and solver parameters
op = opt.Options(dt=0.05, hmin=5e-2, hmax=1., T=2.5, rm=5, Ts=0.5, gradate=False, advect=True,
                 vscale=0.4 if useAdjoint else 0.85)
dt = op.dt
Dt = Constant(dt)
T = op.T
Ts = op.Ts
b = Constant(0.1)
op.checkCFL(b)

# Define inital mesh and FunctionSpace
n = 16
lx = 2 * np.pi
mesh = SquareMesh(n, n, lx, lx)
x, y = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)

# Apply initial condition and define Functions
ic = project(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V.sub(1))
q_ = Function(V)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.assign(ic)
q = Function(V)
u, eta = q.split()
u.rename("Velocity")
eta.rename("Elevation")

# Define Functions to hold residual and adjoint solution data
rho = Function(V)
rho_u, rho_e = rho.split()
rho_u.rename("Velocity residual")
rho_e.rename("Elevation residual")
dual = Function(V)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]            # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Define variational problem
qt = TestFunction(V)
forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

# Initialise counters
t = 0.
cnt = 0

if useAdjoint:
    print('Starting fixed mesh primal run (forwards in time)')
    finished = False
    primalTimer = clock()
    while t < T:
        # Solve problem at current timestep
        forwardSolver.solve()
        q_.assign(q)

        # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
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
        print('t = %.3fs' % t)
        forwardFile.write(u, eta, time=t)
        residualFile.write(rho_u, rho_e, time=t)
        t += dt
        cnt += 1
    cnt -= 1
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    # Set up adjoint problem
    J = form.objectiveFunctionalSW(q, Tstart=Ts, x1=0., x2=np.pi / 2, y1=0.5 * np.pi, y2=1.5 * np.pi, smooth=False)
    parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
    t = T
    save = True

    # Time integrate (backwards)
    print('Starting fixed mesh dual run (backwards in time)')
    dualTimer = clock()
    for (variable, solution) in compute_adjoint(J):
        try:
            if save:
                # Load adjoint data. NOTE the interpolation operator is overloaded
                dual_u.dat.data[:] = variable.dat.data[0]
                dual_e.dat.data[:] = variable.dat.data[1]

                # Load residual data from HDF5
                with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_READ) as loadResidual:
                    loadResidual.load(rho_u)
                    loadResidual.load(rho_e)
                    loadResidual.close()

                # Estimate error using forward residual
                epsilon = assemble(v * inner(rho, dual) * dx)
                epsilon.dat.data[:] = np.abs(epsilon.dat.data) # / assemble(epsilon * dx)
                epsilon.rename("Error indicator")

                # Save error indicator data to HDF5
                if not cnt % rm:
                    with DumbCheckpoint(dirName + 'hdf5/error_' + stor.indexString(cnt), mode=FILE_CREATE) as saveError:
                        saveError.store(epsilon)
                        saveError.close()

                # Print to screen, save data and increment counters
                print('t = %.3fs' % t)
                adjointFile.write(dual_u, dual_e, time=t)
                errorFile.write(epsilon, time=t)
                t -= dt
                cnt -= 1
                save = False
            else:
                save = True
        except:
            continue
    dualTimer = clock() - dualTimer
    print('Adjoint run complete. Run time: %.3fs' % dualTimer)
    t += dt
    cnt += 1

    # Reset initial conditions for primal problem and recreate error indicator placeholder
    u_.interpolate(Expression([0, 0]))
    eta_.assign(ic)
    epsilon = Function(P0, name="Error indicator")

# TODO: adaptive run
