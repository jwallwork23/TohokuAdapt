from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor


# Define initial mesh and mesh statistics placeholders
print('*********************** TOHOKU TSUNAMI SIMULATION *********************\n')
approach = input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': ") or 'simpleAdapt'
useAdjoint = approach == 'goalBased'
op = opt.Options(vscale=0.4 if useAdjoint else 0.85,
                 rm=60 if useAdjoint else 30,
                 gradate=True if useAdjoint else False,
                 advect=False,
                 outputHessian=True,
                 coarseness=5)
mesh, eta0, b = msh.TohokuDomain(op.coarseness)

# Establish filenames
dirName = 'plots/' + approach + '/' + msh.MeshSetup(op.coarseness).meshName + '/'
msh.saveMesh(mesh, dirName + 'hdf5/mesh')
forwardFile = File(dirName + "forward.pvd")
residualFile = File(dirName + "residual.pvd")
adjointFile = File(dirName + "adjointSW.pvd")
errorFile = File(dirName + "errorIndicatorSW.pvd")
adaptiveFile = File(dirName + "goalBasedSW.pvd") if approach == 'goalBased' else File(dirName + "simpleAdaptSW.pvd")
hessianFile = File(dirName + "hessian.pvd")

# Specify physical and solver parameters
dt = op.dt
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
ndump = op.ndump
op.checkCFL(b)

# Define variables of problem and apply initial conditions
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = Function(V)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(eta0)
q = Function(V)
q.assign(q_)
u, eta = q.split()
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]            # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if approach in ('fixedMesh', 'goalBased'):
    # Define variational problem
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    if useAdjoint:
        P0 = FunctionSpace(mesh, "DG", 0)
        v = TestFunction(P0)

        # Define Function to hold residual data
        rho = Function(V)
        rho_u, rho_e = rho.split()
        rho_u.rename("Velocity residual")
        rho_e.rename("Elevation residual")
        dual = Function(V)
        dual_u, dual_e = dual.split()
        dual_u.rename("Adjoint velocity")
        dual_e.rename("Adjoint elevation")

    print('\nStarting fixed mesh primal run (forwards in time)')
    finished = False
    primalTimer = clock()
    forwardFile.write(u, eta, time=t)
    while t < T + dt:
        # Solve problem at current timestep
        forwardSolver.solve()
        q_.assign(q)

        if useAdjoint:
            # Mark timesteps to be used in adjoint simulation
            if t >= T:
                finished = True
            if t == 0.:
                adj_start_timestep()
            else:
                adj_inc_timestep(time=t, finished=finished)

            if not cnt % rm:
                # Approximate residual of forward equation and save to HDF5
                Au, Ae = form.strongResidualSW(q, q_, b, Dt)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_CREATE) as chk:
                    chk.store(rho_u)
                    chk.store(rho_e)
                    chk.close()
                residualFile.write(rho_u, rho_e, time=t)

        if not cnt % ndump:
            forwardFile.write(u, eta, time=t)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    cnt -=1
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

if approach == 'goalBased':
    # Set up adjoint problem
    J = form.objectiveFunctionalSW(q, plot=True)
    parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
    t = T
    save = True

    # Time integrate (backwards)
    print('\nStarting fixed mesh dual run (backwards in time)')
    dualTimer = clock()
    for (variable, solution) in compute_adjoint(J):
        if save:
            # Load adjoint data. NOTE the interpolation operator is overloaded
            dual_u.dat.data[:] = variable.dat.data[0]
            dual_e.dat.data[:] = variable.dat.data[1]

            if not cnt % rm:
                indexStr = stor.indexString(cnt)

                # Load residual data from HDF5
                with DumbCheckpoint(dirName + 'hdf5/residual_' + indexStr, mode=FILE_READ) as loadResidual:
                    loadResidual.load(rho_u)
                    loadResidual.load(rho_e)
                    loadResidual.close()

                # Estimate error using forward residual
                epsilon = assemble(v * inner(rho, dual) * dx)
                epsNorm = np.abs(assemble(inner(rho, dual) * dx))   # Normalise
                if epsNorm == 0.:
                    epsNorm = 1.
                epsilon.dat.data[:] = np.abs(epsilon.dat.data) / epsNorm * 1e6      # TODO: fairly arbitrary scaling
                epsilon.rename("Error indicator")

                # Save error indicator data to HDF5
                with DumbCheckpoint(dirName + 'hdf5/error_' + indexStr, mode=FILE_CREATE) as saveError:
                    saveError.store(epsilon)
                    saveError.close()

                # Print to screen, save data and increment counters
                errorFile.write(epsilon, time=t)

            if not cnt % ndump:
                adjointFile.write(dual_u, dual_e, time=t)
                print('t = %.2fs' % t)
            t -= dt
            cnt -= 1
            save = False
        else:
            save = True
        if (cnt == -1) | (t < -dt):
            break
    dualTimer = clock() - dualTimer
    print('Adjoint run complete. Run time: %.3fs' % dualTimer)
    t += dt
    cnt += 1

    # Reset initial conditions for primal problem and recreate error indicator placeholder
    u_.interpolate(Expression([0, 0]))
    eta_.interpolate(eta0)
    epsilon = Function(P0, name="Error indicator")

if approach in ('simpleAdapt', 'goalBased'):

    if useAdjoint and op.gradate:
        h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))

    print('\nStarting adaptive mesh primal run (forwards in time)')
    adaptTimer = clock()
    while t <= T:
        if not cnt % rm:
            stepTimer = clock()

            # Reconstruct Hessian
            W = TensorFunctionSpace(mesh, "CG", 1)
            H = adap.constructHessian(mesh, W, eta, op=op)

            # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
            if useAdjoint & (cnt != 0):
                with DumbCheckpoint(dirName + 'hdf5/error_' + stor.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)                 # P0 field on the initial mesh
                    loadError.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
                for k in range(mesh.topology.num_vertices()):
                    H.dat.data[k] *= errEst.dat.data[k]     # Scale by error estimate
            if op.outputHessian:
                H.rename("Hessian")
                hessianFile.write(H, time=t)

            # Adapt mesh and interpolate variables
            M = adap.computeSteadyMetric(mesh, W, H, eta, nVerT=nVerT, op=op)
            if op.gradate:
                if useAdjoint:
                    M_ = adap.isotropicMetric(W, inte.interp(mesh, h)[0], bdy=True, op=op)  # Initial boundary metric
                    M = adap.metricIntersection(mesh, W, M, M_, bdy=True)
                adap.metricGradation(mesh, M)
            if op.advect:
                M = adap.advectMetric(M, u, Dt, n=rm)
            mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
            q_ = inte.mixedPairInterp(mesh, V, q_)
            b = inte.interp(mesh, b)[0]     # Combine this in above interpolation for speed
            q = Function(V)
            u, eta = q.split()
            u.rename("Velocity")
            eta.rename("Elevation")

            # Re-establish variational form
            qt = TestFunction(V)
            adaptProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
            adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)

            # Get mesh stats
            nEle = msh.meshStats(mesh)[0]
            N = [min(nEle, N[0]), max(nEle, N[1])]
            Sn += nEle
            op.printToScreen(cnt / rm + 1, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, N)

        # Solve problem at current timestep
        adaptSolver.solve()
        q_.assign(q)

        if not cnt % ndump:
            adaptiveFile.write(u, eta, time=t)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)

# Print to screen timing analyses (and error in pure advection case)
if useAdjoint:
    print("TIMINGS:         Forward run   %5.3fs, Adjoint run   %5.3fs, Adaptive run   %5.3fs" %
          (primalTimer, dualTimer, adaptTimer))
