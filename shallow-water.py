from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.bootstrapping as boot
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt


print('\n*********************** SHALLOW WATER TEST PROBLEM ************************\n')
print('Mesh adaptive solver defined on a square mesh')
approach, getData, getError = msc.cheatCodes(input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': "))
useAdjoint = approach == 'goalBased'
tAdapt = True

# Establish filenames
dirName = "plots/testSuite/"
forwardFile = File(dirName + "forwardSW.pvd")
residualFile = File(dirName + "residualSW.pvd")
adjointFile = File(dirName + "adjointSW.pvd")
errorFile = File(dirName + "errorIndicatorSW.pvd")
adaptiveFile = File(dirName + "goalBasedSW.pvd") if useAdjoint else File(dirName + "simpleAdaptSW.pvd")

# # Establish initial mesh resolution
# bootTimer = clock()
# print('\nBootstrapping to establish optimal mesh resolution')
# n = boot.bootstrap('shallow-water', tol=0.01)[0]
# bootTimer = clock() - bootTimer
# print('Bootstrapping run time: %.3fs\n' % bootTimer)
n = 64

# Define initial Meshes
lx = 2 * np.pi
mesh_H = SquareMesh(n, n, lx, lx)   # Computational mesh
mesh_h = msh.isoP2(mesh_H)          # Finer mesh (h < H) upon which to approximate error
x, y = SpatialCoordinate(mesh_H)

# Define FunctionSpaces and specify physical and solver parameters
op = opt.Options(Tstart=0.5,
                 Tend=2.5,
                 family='dg-cg',
                 hmin=5e-2,
                 hmax=1.,
                 rm=5,
                 gradate=False,
                 advect=False,
                 window=True if useAdjoint else False,
                 vscale=0.4 if useAdjoint else 0.85,
                 plotpvd=True)
V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)
b = Constant(0.1)
H = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
dt = adap.adaptTimestepSW(mesh_H, b)
print('Using initial timestep = %4.3fs\n' % dt)
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
ndump = op.ndump

# Define Functions relating to goalBased approach
if useAdjoint:
    rho = Function(V_h)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    dual = Function(V_H)
    dual_u, dual_e = dual.split()
    dual_h = Function(V_h)
    dual_h_u, dual_h_e = dual_h.split()
    dual_h_u.rename("Adjoint velocity")
    dual_h_e.rename("Adjoint elevation")
    P0_h = FunctionSpace(mesh_h, "DG", 0)
    v = TestFunction(P0_h)
    epsilon = Function(P0_h, name="Error indicator")

# Apply initial condition and define Functions
ic = project(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V.sub(1))
q_ = Function(V_H)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.assign(ic)
q = Function(V_H)
q.assign(q_)
u, eta = q.split()
u.rename("Velocity")
eta.rename("Elevation")

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iStart = int(op.Tstart / dt)
iEnd = int(np.ceil(T / dt))
nEle, nVer = msh.meshStats(mesh_H)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if getData:
    # Define variational problem
    qt = TestFunction(V_H)
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    print('\nStarting primal run (forwards in time) on a fixed mesh with %d elements' % nEle)
    finished = False
    primalTimer = clock()
    if op.plotpvd:
        forwardFile.write(u, eta, time=t)
    while t < T:
        # Solve problem at current timestep
        forwardSolver.solve()

        # Approximate residual of forward equation and save to HDF5
        if useAdjoint:
            if cnt % rm == 0:
                tic = clock()
                qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                print('#### DEBUG: solution pair %d interpolated. Time: %5.1fs' % (int(rm/(cnt+1)), clock()-tic))
                tic = clock()
                Au, Ae = form.strongResidualSW(qh, q_h, b, Dt)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                print('#### DEBUG: residual %d approximated. Time: %5.1fs' % (int(rm/(cnt+1)), clock()-tic))
                tic = clock()
                with DumbCheckpoint(dirName + 'hdf5/residual_SW' + msc.indexString(cnt), mode=FILE_CREATE) as saveRes:
                    saveRes.store(rho_u)
                    saveRes.store(rho_e)
                    saveRes.close()
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=t)
                print('#### DEBUG: residual %d stored. Time: %5.1fs' % (int(rm/(cnt+1)), clock()-tic))

        # Update solution at previous timestep
        q_.assign(q)

        # Mark timesteps to be used in adjoint simulation
        tic = clock()
        if useAdjoint:
            if t >= T - dt:
                finished = True
            if t == 0.:
                adj_start_timestep()
            else:
                adj_inc_timestep(time=t, finished=finished)
            print('#### DEBUG: solution data logged for timestep %d. Time: %5.1fs' % (cnt, clock()-tic))

        if op.plotpvd & (cnt % ndump == 0):
            forwardFile.write(u, eta, time=t)
        print('t = %.3fs' % t)
        t += dt
        cnt += 1
    cnt -= 1
    cntT = cnt      # Total number of steps
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    if useAdjoint:
        # Set up adjoint problem
        J = form.objectiveFunctionalSW(q, Tstart=Ts, x1=0., x2=np.pi/2, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)
        parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
        save = True

        # Time integrate (backwards)
        print('\nStarting fixed mesh dual run (backwards in time)')
        dualTimer = clock()
        for (variable, solution) in compute_adjoint(J):
            if save:
                # Load adjoint data
                dual.assign(variable, annotate=False)
                dual_h = inte.mixedPairInterp(mesh_h, V_h, dual)[0]
                dual_h_u, dual_h_e = dual_h.split()
                dual_h_u.rename('Adjoint velocity')
                dual_h_e.rename('Adjoint elevation')

                # Save adjoint data to HDF5
                if cnt % rm == 0:
                    with DumbCheckpoint(dirName+'hdf5/adjoint_SW'+msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
                        saveAdj.store(dual_h_u)
                        saveAdj.store(dual_h_e)
                        saveAdj.close()
                    print('Adjoint simulation %.2f%% complete' % ((cntT-cnt)/cntT) * 100)
                cnt -= 1
                save = False
            else:
                save = True
            if cnt == -1:
                break
        dualTimer = clock() - dualTimer
        print('Adjoint run complete. Run time: %.3fs' % dualTimer)
        cnt += 1

        # Reset initial conditions for primal problem
        u_.interpolate(Expression([0, 0]))
        eta_.assign(ic)

# Loop back over times to generate error estimators
if getError:
    errorTimer = clock()
    for k in range(0, iEnd, rm):
        print('Generating error estimate %d / %d' % (k / rm + 1, iEnd / rm))
        indexStr = msc.indexString(k)

        # Load residual and adjoint data from HDF5
        with DumbCheckpoint(dirName + 'hdf5/residual_SW' + indexStr, mode=FILE_READ) as loadRes:
            loadRes.load(rho_u)
            loadRes.load(rho_e)
            loadRes.close()
        with DumbCheckpoint(dirName + 'hdf5/adjoint_SW' + indexStr, mode=FILE_READ) as loadAdj:
            loadAdj.load(dual_h_u)
            loadAdj.load(dual_h_e)
            loadAdj.close()

        # Estimate error using dual weighted residual
        epsilon = err.DWR(rho, dual_h, v)      # Currently a P0 field
        # TODO: include functionality for other error estimators

        # Loop over relevant time window
        if op.window:
            for i in range(iStart, cnt, rm):
                with DumbCheckpoint(dirName + 'hdf5/adjoint_SW' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_h_u)
                    loadAdj.load(dual_h_e)
                    loadAdj.close()
                epsilon_ = err.DWR(rho, dual_h, v)
                for j in range(len(epsilon.dat.data)):
                    epsilon.dat.data[j] = max(epsilon.dat.data[j], epsilon_.dat.data[j])

        # Normalise error estimate
        epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)
        epsilon.rename("Error indicator")

        # Store error estimates
        with DumbCheckpoint(dirName + 'hdf5/error_SW' + indexStr, mode=FILE_CREATE) as saveErr:
            saveErr.store(epsilon)
            saveErr.close()
        if op.plotpvd:
            errorFile.write(epsilon, time=t)
    errorTimer = clock() - errorTimer
    print('Errors estimated. Run time: %.3fs' % errorTimer)

if approach in ('simpleAdapt', 'goalBased'):
    t = 0.
    print('\nStarting adaptive mesh primal run (forwards in time)')
    adaptTimer = clock()
    while t <= T:
        if (cnt % rm == 0) & (np.abs(t-T) > 0.5 * dt):
            stepTimer = clock()

            # Construct metric
            W = TensorFunctionSpace(mesh_H, "CG", 1)
            if useAdjoint:
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_SW' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                H = adap.constructHessian(mesh_H, W, eta, op=op)
                M = adap.computeSteadyMetric(mesh_H, W, H, eta, nVerT=nVerT, op=op)
            if op.gradate:
                adap.metricGradation(mesh_H, M)
            if op.advect:
                M = adap.advectMetric(M, u, 2*Dt, n=3*rm)
                # TODO: isotropic advection?

            # Adapt mesh and interpolate variables
            mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
            V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
            q_ = inte.mixedPairInterp(mesh_H, V_H, q_)[0]
            q = Function(V_H)
            u, eta = q.split()
            u.rename("Velocity")
            eta.rename("Elevation")

            # Re-establish variational form
            qt = TestFunction(V)
            adaptProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
            adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)

            if tAdapt:
                dt = adap.adaptTimestepSW(mesh_H, b)
                Dt.assign(dt)

            # Get mesh stats
            nEle = msh.meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt/rm+1, clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, t, dt)

        # Solve problem at current timestep
        adaptSolver.solve()
        q_.assign(q)

        # Print to screen, save data and increment counters
        print('t = %.3fs' % t)
        if op.plotpvd & (cnt % ndump == 0):
            adaptiveFile.write(u, eta, time=t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)

# Print to screen timing analyses
if getData and useAdjoint:
    msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer=bootTimer)