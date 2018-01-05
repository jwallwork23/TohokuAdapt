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
import utils.timeseries as tim


print('*********************** TOHOKU TSUNAMI SIMULATION *********************\n')
approach, getData, getError = msc.cheatCodes(input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': "))
useAdjoint = approach == 'goalBased'
tAdapt = False
bootstrap = False

# Define initial mesh and mesh statistics placeholders
op = opt.Options(vscale=0.045 if useAdjoint else 0.85,
                 rm=60 if useAdjoint else 30,
                 gradate=True if useAdjoint else False,
                 advect=False,
                 window=True,
                 outputHessian=False,
                 plotpvd=True,
                 coarseness=3,
                 gauges=True,
                 ndump=10,
                 mtype='f',
                 iso=True if useAdjoint else False)

# Establish initial mesh resolution
if bootstrap:
    bootTimer = clock()
    print('\nBootstrapping to establish optimal mesh resolution')
    i = boot.bootstrap('firedrake-tsunami', tol=2e10)[0]
    bootTimer = clock() - bootTimer
    print('Bootstrapping run time: %.3fs\n' % bootTimer)
else:
    i = 1
nEle = op.meshes[i]

# Establish filenames
dirName = 'plots/firedrake-tsunami/'
if op.plotpvd:
    forwardFile = File(dirName + "forward.pvd")
    residualFile = File(dirName + "residual.pvd")
    adjointFile = File(dirName + "adjoint.pvd")
    errorFile = File(dirName + "errorIndicator.pvd")
    adaptiveFile = File(dirName + "goalBased.pvd") if useAdjoint else File(dirName + "simpleAdapt.pvd")
if op.outputHessian:
    hessianFile = File(dirName + "hessian.pvd")

# Load Meshes
mesh, eta0, b = msh.TohokuDomain(nEle)
if useAdjoint:
    try:
        assert op.coarseness != 1
    except:
        raise NotImplementedError("Requested mesh resolution not yet available.")
    # Get finer mesh and associated bathymetry
    mesh_N, b_N = msh.TohokuDomain(op.meshes[i+1])[0::2]
    V_N = VectorFunctionSpace(mesh_N, op.space1, op.degree1) * FunctionSpace(mesh_N, op.space2, op.degree2)

# Specify physical and solver parameters
dt = adap.adaptTimestepSW(mesh, b)
print('Using initial timestep = %4.3fs\n' % dt)
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
ndump = op.ndump
op.checkCFL(b)

# Define variables of problem and apply initial conditions
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = Function(V)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]), annotate=False)
eta_.interpolate(eta0, annotate=False)
q = Function(V)
q.assign(q_)
u, eta = q.split()
u.rename("uv_2d")
eta.rename("elev_2d")

# Get initial gauge values
gaugeData = {}
gauges = ("P02", "P06")
v0 = {}
for gauge in gauges:
    v0[gauge] = float(eta.at(op.gaugeCoord(gauge)))

# Define Functions relating to goalBased approach
if useAdjoint:
    rho = Function(V_N)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_N = Function(V_N)
    dual_N_u, dual_N_e = dual_N.split()
    dual_N_u.rename("Adjoint velocity")
    dual_N_e.rename("Adjoint elevation")
    P0_N = FunctionSpace(mesh_N, "DG", 0)
    v = TestFunction(P0_N)
    epsilon = Function(P0_N, name="Error indicator")
    J = form.objectiveFunctionalSW(q, plot=True)

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iStart = int(op.Tstart / dt)    # TODO: alter for t-adapt
iEnd = int(np.ceil(T / dt))     # TODO: alter for t-adapt
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = msh.meshStats(mesh)[1] * op.vscale    # Target #Vertices
nVerT0 = nVerT

# Initialise counters
t = 0.
cnt = 0
save = True

if getData:
    # Define variational problem
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    print('Starting fixed mesh primal run (forwards in time)')
    finished = False
    primalTimer = clock()
    if op.plotpvd:
        forwardFile.write(u, eta, time=t)
    while t < T + dt:
        # Solve problem at current timestep
        forwardSolver.solve()

        # Approximate residual of forward equation and save to HDF5
        if useAdjoint:
            if cnt % rm == 0:
                qN, q_N = inte.mixedPairInterp(mesh_N, V_N, q, q_)
                Au, Ae = form.strongResidualSW(qN, q_N, b_N, Dt)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                with DumbCheckpoint(dirName + 'hdf5/residual_' + msc.indexString(cnt), mode=FILE_CREATE) as saveRes:
                    saveRes.store(rho_u)
                    saveRes.store(rho_e)
                    saveRes.close()
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=t)

        # Update solution at previous timestep
        q_.assign(q)

        # Mark timesteps to be used in adjoint simulation
        if useAdjoint:
            if t >= T:
                finished = True
            if t == 0.:
                adj_start_timestep()
            else:
                adj_inc_timestep(time=t, finished=finished)

        if cnt % ndump == 0:
            if op.plotpvd & (cnt % ndump == 0):
                forwardFile.write(u, eta, time=t)
            if op.gauges and not useAdjoint:
                gaugeData = tim.extractTimeseries(gauges, eta, t, gaugeData, v0, op=op)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    cnt -=1
    cntT = cnt  # Total number of steps
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    if useAdjoint:
        parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
        print('\nStarting fixed mesh dual run (backwards in time)')
        dualTimer = clock()
        for (variable, solution) in compute_adjoint(J):
            if save:
                # Load adjoint data
                dual.assign(variable, annotate=False)
                dual_N = inte.mixedPairInterp(mesh_N, V_N, dual)[0]
                dual_N_u, dual_N_e = dual_N.split()
                dual_N_u.rename('Adjoint velocity')
                dual_N_e.rename('Adjoint elevation')

                # Save adjoint data to HDF5
                if cnt % rm == 0:
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_' + msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
                        saveAdj.store(dual_N_u)
                        saveAdj.store(dual_N_e)
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

        # Reset initial conditions for primal problem and recreate error indicator placeholder
        u_.interpolate(Expression([0, 0]))
        eta_.interpolate(eta0)

# Loop back over times to generate error estimators
if getError:
    errorTimer = clock()
    errEstMean = 0
    for k in range(0, iEnd, rm):
        print('Generating error estimate %d / %d' % (k / rm + 1, iEnd / rm + 1))
        indexStr = msc.indexString(k)

        # Load residual and adjoint data from HDF5
        with DumbCheckpoint(dirName + 'hdf5/residual_' + indexStr, mode=FILE_READ) as loadRes:
            loadRes.load(rho_u)
            loadRes.load(rho_e)
            loadRes.close()
        with DumbCheckpoint(dirName + 'hdf5/adjoint_' + indexStr, mode=FILE_READ) as loadAdj:
            loadAdj.load(dual_N_u)
            loadAdj.load(dual_N_e)
            loadAdj.close()

        # Estimate error using dual weighted residual
        epsilon = err.DWR(rho, dual_N, v)      # Currently a P0 field
        # TODO: include functionality for other error estimators

        # Loop over relevant time window
        if op.window:
            for i in range(max(iStart, cnt), min(iEnd, cnt), rm):
                with DumbCheckpoint(dirName + 'hdf5/adjoint_' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_N_u)
                    loadAdj.load(dual_N_e)
                    loadAdj.close()
                epsilon_ = err.DWR(rho, dual_N, v)
                for j in range(len(epsilon.dat.data)):
                    epsilon.dat.data[j] = max(epsilon.dat.data[j], epsilon_.dat.data[j])
        epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)  # Normalise
        epsilon.rename("Error indicator")

        # Store error estimates
        with DumbCheckpoint(dirName + 'hdf5/error_' + indexStr, mode=FILE_CREATE) as saveErr:
            saveErr.store(epsilon)
            saveErr.close()
        if op.plotpvd:
            errorFile.write(epsilon, time=float(k))
    errorTimer = clock() - errorTimer
    print('Errors estimated. Run time: %.3fs' % errorTimer)

if approach in ('simpleAdapt', 'goalBased'):
    t = 0.
    if useAdjoint & op.gradate:
        h0 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    print('\nStarting adaptive mesh primal run (forwards in time)')
    adaptTimer = clock()
    while t <= T:
        if cnt % rm == 0:      # TODO: change this condition for t-adaptivity?
            stepTimer = clock()

            # Construct metric
            W = TensorFunctionSpace(mesh, "CG", 1)
            if useAdjoint:
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                if op.mtype != 's':
                    if op.iso:
                        M = adap.isotropicMetric(W, eta, op=op)
                    else:
                        H = adap.constructHessian(mesh, W, eta, op=op)
                        M = adap.computeSteadyMetric(mesh, W, H, eta, nVerT=nVerT, op=op)
                if op.mtype != 'f':
                    spd = Function(FunctionSpace(mesh, 'DG', 1)).interpolate(sqrt(dot(u, u)))
                    if op.iso:
                        M2 = adap.isotropicMetric(W, spd, op=op)
                    else:
                        H = adap.constructHessian(mesh, W, spd, op=op)
                        M2 = adap.computeSteadyMetric(mesh, W, H, spd, nVerT=nVerT, op=op)
                    M = adap.metricIntersection(mesh, W, M, M2) if op.mtype == 'b' else M2
            if op.gradate:
                if useAdjoint:
                    M_ = adap.isotropicMetric(W, inte.interp(mesh, h0)[0], bdy=True, op=op) # Initial boundary metric
                    M = adap.metricIntersection(mesh, W, M, M_, bdy=True)
                adap.metricGradation(mesh, M)
                # TODO: always gradate to coast
            if op.advect:
                M = adap.advectMetric(M, u, 2*Dt, n=3*rm)
                # TODO: isotropic advection?

            # Adapt mesh and interpolate variables
            mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
            q_ = inte.mixedPairInterp(mesh, V, q_)[0]
            b = inte.interp(mesh, b)[0]     # TODO: Combine this in above interpolation for speed
            q = Function(V)
            u, eta = q.split()
            u.rename("uv_2d")
            eta.rename("elev_2d")

            # Re-establish variational form
            qt = TestFunction(V)
            adaptProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
            adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)

            if tAdapt:
                dt = adap.adaptTimestepSW(mesh, b)
                Dt.assign(dt)

            # Get mesh stats
            nEle = msh.meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt/rm+1, clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, t, dt)

        # Solve problem at current timestep
        adaptSolver.solve()
        q_.assign(q)

        if cnt % ndump == 0:
            if op.plotpvd:
                adaptiveFile.write(u, eta, time=t)
            if op.gauges:
                gaugeData = tim.extractTimeseries(gauges, eta, t, gaugeData, v0, op=op)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)

# Print to screen timing analyses
if getData and useAdjoint:
    msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)

# Save and plot timeseries
name = input("Enter a name for these time series (e.g. 'goalBased8-12-17'): ") or 'test'
for gauge in gauges:
    tim.saveTimeseries(gauge, gaugeData, name=name)
