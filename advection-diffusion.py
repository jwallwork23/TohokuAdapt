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
mesh_H = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
mesh_h = msh.isoP2(mesh_H)              # Finer mesh (h < H) upon which to approximate error
x, y = SpatialCoordinate(mesh_H)

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
V_H = FunctionSpace(mesh_H, "CG", 2)
V_h = FunctionSpace(mesh_h, "CG", 2)
w = Function(VectorFunctionSpace(mesh_H, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
dt = adap.adaptTimestepAD(w)
print('Using initial timestep = %4.3fs\n' % dt)
Dt = Constant(dt)
T = op.Tend
ndump = op.ndump
nu = 1e-3 if diffusion else 0.

# Define Functions relating to goalBased approach
if useAdjoint:
    dual = Function(V_H, name='Adjoint')
    dual_h = Function(V_h, name='Fine mesh adjoint')
    rho = Function(V_h, name='Residual')
    P0_h = FunctionSpace(mesh_h, "DG", 0)
    v = TestFunction(P0_h)
    epsilon = Function(P0_h, name="Error indicator")

# Apply initial condition and define Functions
ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V_H)
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
phi_next = Function(V_H, name='Concentration next')

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iEnd = np.ceil(T / dt)
nEle, nVer = msh.meshStats(mesh_H)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if getData:
    # Define variational problem
    psi = TestFunction(V_H)
    F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)

    print('\nStarting primal run (forwards in time) on a fixed mesh with %d elements' % nEle)
    finished = False
    primalTimer = clock()
    if op.plotpvd:
        forwardFile.write(phi, time=t)
    while t < T:
        # Solve problem at current timestep
        solve(F == 0, phi_next)

        # Approximate residual of forward equation and save to HDF5
        if useAdjoint:
            if cnt % rm == 0:
                phi_next_h, phi_h, w_h = inte.interp(mesh_h, phi_next, phi, w)
                rho.interpolate(form.strongResidualAD(phi_next_h, phi_h, w_h, Dt, nu=nu))
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
                dual_h = inte.interp(mesh_h, dual)[0]
                dual_h.rename('Fine mesh adjoint')

                # Save adjoint data to HDF5
                if cnt % rm == 0:
                    with DumbCheckpoint(dirName+'hdf5/adjoint_AD'+msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
                        saveAdj.store(dual_h)
                        saveAdj.close()
                    print('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT) * 100)
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
            loadAdj.load(dual_h)
            loadAdj.close()

        # Estimate error using dual weighted residual
        epsilon = err.DWR(rho, dual_h, v)  # Currently a P0 field
        # TODO: include functionality for other error estimators

        # Loop over relevant time window
        if op.window:
            for i in range(0, cnt, rm):
                with DumbCheckpoint(dirName + 'hdf5/adjoint_AD' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_h)
                    loadAdj.close()
                epsilon_ = err.DWR(rho, dual_h, v)
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
            W = TensorFunctionSpace(mesh_H, "CG", 1)
            if useAdjoint:
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_AD' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                H = adap.constructHessian(mesh_H, W, phi, op=op)
                M = adap.computeSteadyMetric(mesh_H, W, H, phi, nVerT=nVerT, op=op)
            if op.gradate:
                adap.metricGradation(mesh_H, M)
            if op.advect:
                if useAdjoint:
                    M = adap.isotropicAdvection(M, errEst, w, 2*Dt, n=3*rm)
                else:
                    M = adap.advectMetric(M, w, 2*Dt, n=3*rm)

            # Adapt mesh and interpolate variables
            mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
            phi = inte.interp(mesh_H, phi)[0]
            phi.rename("Concentration")
            V_H = FunctionSpace(mesh_H, "CG", 2)
            phi_next = Function(V_H)

            # Re-establish bilinear form and set boundary conditions
            psi = TestFunction(V_H)
            w = Function(VectorFunctionSpace(mesh_H, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
            F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)

            if tAdapt:
                dt = adap.adaptTimestepAD(w)
                Dt.assign(dt)

            # Get mesh stats
            nEle = msh.meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt / rm + 1, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, mM, t, dt)

        # Solve problem at current timestep
        solve(F == 0, phi_next)
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