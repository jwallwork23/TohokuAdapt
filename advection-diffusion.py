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


print('\n******************** ADVECTION-DIFFUSION TEST PROBLEM *********************\n')
print('Mesh adaptive solver initially defined on a rectangular mesh')
approach, getData, getError = msc.cheatCodes(input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': "))
useAdjoint = approach == 'goalBased'
diffusion = True
tAdapt = True

# Establish filenames
dirName = "plots/testSuite/"
forwardFile = File(dirName + "forwardAD.pvd")
residualFile = File(dirName + "residualAD.pvd")
adjointFile = File(dirName + "adjointAD.pvd")
errorFile = File(dirName + "errorIndicatorAD.pvd")
adaptiveFile = File(dirName + "goalBasedAD.pvd") if useAdjoint else File(dirName + "simpleAdaptAD.pvd")

# Establish initial mesh resolution
bootTimer = clock()
print('\nBootstrapping to establish optimal mesh resolution')
n = boot.bootstrap('advection-diffusion', tol=0.01)[0]
bootTimer = clock() - bootTimer
print('Bootstrapping run time: %.3fs\n' % bootTimer)

# Define initial Meshes
N = 2 * n
mesh = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
mesh_N = RectangleMesh(4 * N, N, 4, 1)  # Finer mesh (N > n) upon which to approximate error
x, y = SpatialCoordinate(mesh)

# Define FunctionSpaces and specify physical and solver parameters
op = opt.Options(dt=0.04,
                 Tend=2.4,
                 hmin=5e-2,
                 hmax=0.8,
                 rm=5,
                 gradate=False,
                 advect=False,
                 window=True,
                 vscale=0.4 if useAdjoint else 0.85,
                 plotpvd=True if getData == False else False)
V = FunctionSpace(mesh, "CG", 2)
V_N = FunctionSpace(mesh_N, "CG", 2)
w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
dt = adap.adaptTimestepAD(w)
print('Using initial timestep = %4.3fs\n' % dt)
Dt = Constant(dt)
T = op.Tend
ndump = op.ndump
nu = 1e-3 if diffusion else 0.

# Define Functions relating to goalBased approach
if useAdjoint:
    dual = Function(V, name='Adjoint')
    dual_N = Function(V_N, name='Fine mesh adjoint')
    rho = Function(V_N, name='Residual')
    P0_N = FunctionSpace(mesh_N, "DG", 0)
    v = TestFunction(P0_N)
    epsilon = Function(P0_N, name="Error indicator")

# Apply initial condition and define Functions
ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
phi_next = Function(V, name='Concentration next')

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iEnd = np.ceil(T / dt)
nEle, nVer = msh.meshStats(mesh)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if getData:
    # Define variational problem
    psi = TestFunction(V)
    F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
    bc = DirichletBC(V, 0., "on_boundary")

    print('\nStarting primal run (forwards in time) on a fixed mesh with %d elements' % nEle)
    finished = False
    primalTimer = clock()
    if op.plotpvd:
        forwardFile.write(phi, time=t)
    while t < T:
        # Solve problem at current timestep
        solve(F == 0, phi_next, bc)

        # Approximate residual of forward equation and save to HDF5
        if useAdjoint:
            if cnt % rm == 0:
                phi_next_N, phi_N, w_N = inte.interp(mesh_N, phi_next, phi, w)
                rho.interpolate(form.strongResidualAD(phi_next_N, phi_N, w_N, Dt, nu=nu))
                with DumbCheckpoint(dirName + 'hdf5/residual_AD' + msc.indexString(cnt), mode=FILE_CREATE) as saveRes:
                    saveRes.store(rho)
                    saveRes.close()
                if op.plotpvd:
                    residualFile.write(rho, time=t)

        # Update solution at previous timestep
        phi.assign(phi_next)

        # Mark timesteps to be used in adjoint simulation
        if t > T:
            finished = True
        if useAdjoint:
            if t == 0.:
                adj_start_timestep()
            else:
                adj_inc_timestep(time=t, finished=finished)

        if op.plotpvd & (cnt % ndump == 0):
            forwardFile.write(phi, time=t)
        print('t = %.3fs' % t)
        t += dt
        cnt += 1
    cnt -= 1
    cntT = cnt  # Total number of steps
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    if useAdjoint:
        # Set up adjoint problem
        J = form.objectiveFunctionalAD(phi)
        parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
        save = True

        # Time integrate (backwards)
        print('\nStarting fixed mesh dual run (backwards in time)')
        dualTimer = clock()
        for (variable, solution) in compute_adjoint(J):
            if save:
                # Load adjoint data
                dual.interpolate(variable, annotate=False)
                dual_N = inte.interp(mesh_N, dual)[0]
                dual_N.rename('Fine mesh adjoint')

                # Save adjoint data to HDF5
                if cnt % rm == 0:
                    with DumbCheckpoint(dirName+'hdf5/adjoint_AD'+msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
                        saveAdj.store(dual_N)
                        saveAdj.close()
                print('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT))
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
        phi = ic.copy(deepcopy=True)
        phi.rename('Concentration')

# Loop back over times to generate error estimators
if getError:
    errorTimer = clock()
    for k in range(0, iEnd, rm):
        print('Generating error estimate %d / %d' % (k / rm + 1, iEnd / rm))
        indexStr = msc.indexString(k)

        # Load residual and adjoint data from HDF5
        with DumbCheckpoint(dirName + 'hdf5/residual_AD' + indexStr, mode=FILE_READ) as loadRes:
            loadRes.load(rho)
            loadRes.close()
        with DumbCheckpoint(dirName + 'hdf5/adjoint_AD' + indexStr, mode=FILE_READ) as loadAdj:
            loadAdj.load(dual_N)
            loadAdj.close()

        # Estimate error using dual weighted residual
        epsilon = err.DWR(rho, dual_N, v)  # Currently a P0 field
        # TODO: include functionality for other error estimators

        # Loop over relevant time window
        if op.window:
            for i in range(0, cnt, rm):
                with DumbCheckpoint(dirName + 'hdf5/adjoint_AD' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_N)
                    loadAdj.close()
                epsilon_ = err.DWR(rho, dual_N, v)
                for j in range(len(epsilon.dat.data)):
                    epsilon.dat.data[j] = max(epsilon.dat.data[j], epsilon_.dat.data[j])

        # Normalise error estimate
        epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)
        epsilon.rename("Error indicator")

        # Store error estimates
        with DumbCheckpoint(dirName + 'hdf5/error_AD' + indexStr, mode=FILE_CREATE) as saveErr:
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
            W = TensorFunctionSpace(mesh, "CG", 1)
            if useAdjoint:
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_AD' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                H = adap.constructHessian(mesh, W, phi, op=op)
                M = adap.computeSteadyMetric(mesh, W, H, phi, nVerT=nVerT, op=op)
            if op.gradate:
                adap.metricGradation(mesh, M)
            if op.advect:
                if useAdjoint:
                    M = adap.isotropicAdvection(M, errEst, w, 2*Dt, n=3*rm)
                else:
                    M = adap.advectMetric(M, w, 2*Dt, n=3*rm)

            # Adapt mesh and interpolate variables
            mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            phi = inte.interp(mesh, phi)[0]
            phi.rename("Concentration")
            V = FunctionSpace(mesh, "CG", 2)
            phi_next = Function(V)

            # Re-establish bilinear form and set boundary conditions
            psi = TestFunction(V)
            w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
            F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
            bc = DirichletBC(V, 0., "on_boundary")

            if tAdapt:
                dt = adap.adaptTimestepAD(w)
                Dt.assign(dt)

            # Get mesh stats
            nEle = msh.meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt / rm + 1, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, mM, t, dt)

        # Solve problem at current timestep
        solve(F == 0, phi_next, bc)
        phi.assign(phi_next)

        # Print to screen, save data and increment counters
        print('t = %.3fs' % t)
        if op.plotpvd & (cnt % ndump == 0):
            adaptiveFile.write(phi, time=t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)

# Print to screen timing analyses
if getData and useAdjoint:
    msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer=bootTimer)
