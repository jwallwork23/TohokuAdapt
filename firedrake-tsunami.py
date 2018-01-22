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
approach, getData, getError, useAdjoint = msc.cheatCodes(input(
    "Choose error estimator: 'hessianBased', 'explicit', 'adjointBased' or 'goalBased': "))
tAdapt = False
bootstrap = False
outputOF = True

# Define initial mesh and mesh statistics placeholders
op = opt.Options(vscale=0.4 if useAdjoint else 0.85,
                 rm=60 if useAdjoint else 30,
                 # rm=60,
                 gradate=True if (useAdjoint or approach == 'explicit') else False,
                 advect=False,
                 window=True if approach == 'adjointBased' else False,
                 outputHessian=False,
                 plotpvd=True,
                 gauges=False,
                 ndump=10,
                 mtype='s',
                 iso=False)

# Establish initial mesh resolution
if bootstrap:
    bootTimer = clock()
    print('\nBootstrapping to establish optimal mesh resolution')
    i = boot.bootstrap('firedrake-tsunami', tol=2e10)[0]
    bootTimer = clock() - bootTimer
    print('Bootstrapping run time: %.3fs\n' % bootTimer)
else:
    i = 2
nEle = op.meshes[i]

# Establish filenames
dirName = 'plots/firedrake-tsunami/'
if op.plotpvd:
    forwardFile = File(dirName + "forward.pvd")
    residualFile = File(dirName + "residual.pvd")
    errorFile = File(dirName + "errorIndicator.pvd")
adaptiveFile = File(dirName + approach + ".pvd")
if op.outputHessian:
    hessianFile = File(dirName + "hessian.pvd")

# Load Mesh(es)
mesh_H, eta0, b = msh.TohokuDomain(nEle)        # Computational mesh
if approach in ('explicit', 'goalBased'):
    mesh_h = adap.isoP2(mesh_H)                  # Finer mesh (h < H) upon which to approximate error
    b_h = msh.TohokuDomain(mesh=mesh_h)[2]
    V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)

# Specify physical and solver parameters
dt = adap.adaptTimestepSW(mesh_H, b)
print('Using initial timestep = %4.3fs\n' % dt)
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
ndump = op.ndump
op.checkCFL(b)

# Define variables of problem and apply initial conditions
V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
q_ = Function(V_H)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]), annotate=False)
eta_.interpolate(eta0, annotate=False)
q = Function(V_H)
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
if approach in ('explicit', 'goalBased'):
    rho = Function(V_h)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    if useAdjoint:
        dual_h = Function(V_h)
        dual_h_u, dual_h_e = dual_h.split()
        dual_h_u.rename('Fine adjoint velocity')
        dual_h_e.rename('Fine adjoint elevation')
    else:
        qh = Function(V_h)
        uh, eh = qh.split()
        uh.rename("Fine velocity")
        eh.rename("Fine elevation")
if useAdjoint:
    dual = Function(V_H)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
    J = form.objectiveFunctionalSW(q, plot=True)
if approach in ('explicit', 'adjointBased', 'goalBased'):
    P0 = FunctionSpace(mesh_H, "DG", 0) if approach == 'adjointBased' else FunctionSpace(mesh_h, "DG", 0)
    v = TestFunction(P0)
    epsilon = Function(P0, name="Error indicator")

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
if tAdapt:
    raise NotImplementedError("Mesh adaptive routines not quite calibrated for t-adaptivity")
else:
    iStart = int(op.Tstart / dt)
    iEnd = int(np.ceil(T / dt))
mM = [nEle, nEle]                               # Min/max #Elements
Sn = nEle
nVerT = msh.meshStats(mesh_H)[1] * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0
save = True

if getData:
    # Define variational problem
    qt = TestFunction(V_H)
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
        if cnt % rm == 0:
            if approach in ('explicit', 'goalBased'):
                qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                Au, Ae = form.strongResidualSW(qh, q_h, b_h, Dt)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                with DumbCheckpoint(dirName + 'hdf5/residual_' + msc.indexString(cnt), mode=FILE_CREATE) as saveRes:
                    saveRes.store(rho_u)
                    saveRes.store(rho_e)
                    if approach == 'explicit':
                        uh, eh = qh.split()
                        saveRes.store(uh)
                        saveRes.store(eh)
                    saveRes.close()
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=t)
            if approach in ('explicit', 'adjointBased'):
                with DumbCheckpoint(dirName + 'hdf5/forward_' + msc.indexString(cnt), mode=FILE_CREATE) as saveFor:
                    saveFor.store(u)
                    saveFor.store(eta)
                    saveFor.close()

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
            if op.plotpvd:
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
                # Load adjoint data and save to HDF5
                if cnt % rm == 0:
                    indexStr = msc.indexString(cnt)
                    dual.assign(variable, annotate=False)
                    if approach == 'adjointBased':
                        dual_u, dual_e = dual.split()
                        dual_u.rename('Adjoint velocity')
                        dual_e.rename('Adjoint elevation')
                        with DumbCheckpoint(dirName + 'hdf5/adjoint_H_' + indexStr, mode=FILE_CREATE) as saveAdjH:
                            saveAdjH.store(dual_u)
                            saveAdjH.store(dual_e)
                            saveAdjH.close()
                    else:
                        dual_h = inte.mixedPairInterp(mesh_h, V_h, dual)[0]
                        dual_h_u, dual_h_e = dual_h.split()
                        dual_h_u.rename('Fine adjoint velocity')
                        dual_h_e.rename('Fine adjoint elevation')
                        with DumbCheckpoint(dirName + 'hdf5/adjoint_' + indexStr, mode=FILE_CREATE) as saveAdj:
                            saveAdj.store(dual_h_u)
                            saveAdj.store(dual_h_e)
                            saveAdj.close()
                    print('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT * 100))
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
    print('\nStarting error estimate generation')
    errorTimer = clock()
    for k in range(0, iEnd, rm):
        print('Generating error estimate %d / %d' % (k / rm + 1, iEnd / rm + 1))
        indexStr = msc.indexString(k)

        # Load residual and adjoint data from HDF5
        if useAdjoint:
            if approach == 'goalBased':
                with DumbCheckpoint(dirName + 'hdf5/adjoint_' + indexStr, mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_h_u)
                    loadAdj.load(dual_h_e)
                    loadAdj.close()
            else:
                with DumbCheckpoint(dirName + 'hdf5/adjoint_H_' + indexStr, mode=FILE_READ) as loadAdjH:
                    loadAdjH.load(dual_u)
                    loadAdjH.load(dual_e)
                    loadAdjH.close()
        if approach in ('explicit', 'goalBased'):
            with DumbCheckpoint(dirName + 'hdf5/residual_' + indexStr, mode=FILE_READ) as loadRes:
                loadRes.load(rho_u)
                loadRes.load(rho_e)
                loadRes.close()
        if approach in ('explicit', 'adjointBased'):
            with DumbCheckpoint(dirName + 'hdf5/forward_' + indexStr, mode=FILE_READ) as loadFor:
                loadFor.load(u)
                loadFor.load(eta)
                loadFor.close()
        if approach == 'adjointBased':
            epsilon = err.basicErrorEstimator(q, dual, v)
        elif approach == 'goalBased':
            epsilon = err.DWR(rho, dual_h, v)
        elif approach == 'explicit':
            epsilon = err.explicitErrorEstimator(q, rho, v)

        # Loop over relevant time window
        if op.window:
            for i in range(cnt, min(cnt+iEnd-iStart, iEnd), rm):
                if approach == 'goalBased':
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_h_u)
                        loadAdj.load(dual_h_e)
                        loadAdj.close()
                    epsilon_ = err.DWR(rho, dual_h, v)
                elif approach == 'adjointBased':
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_H_' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_u)
                        loadAdj.load(dual_e)
                        loadAdj.close()
                    epsilon_ = err.basicErrorEstimator(q, dual, v)
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

if approach in ('hessianBased', 'explicit', 'adjointBased', 'goalBased'):
    t = 0.
    J_trap = 0.
    started = False
    if op.gradate:
        H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
    print('\nStarting adaptive mesh primal run (forwards in time)')
    adaptTimer = clock()
    while t <= T:
        if cnt % rm == 0:      # TODO: change this condition for t-adaptivity?
            stepTimer = clock()

            # Construct metric
            W = TensorFunctionSpace(mesh_H, "CG", 1)
            if approach in ('explicit', 'adjointBased', 'goalBased'):
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                if op.mtype != 's':
                    if op.iso:
                        M = adap.isotropicMetric(W, eta, op=op)
                    else:
                        H = adap.constructHessian(mesh_H, W, eta, op=op)
                        M = adap.computeSteadyMetric(mesh_H, W, H, eta, nVerT=nVerT, op=op)
                if op.mtype != 'f':
                    spd = Function(FunctionSpace(mesh_H, 'DG', 1)).interpolate(sqrt(dot(u, u)))
                    if op.iso:
                        M2 = adap.isotropicMetric(W, spd, op=op)
                    else:
                        H = adap.constructHessian(mesh_H, W, spd, op=op)
                        M2 = adap.computeSteadyMetric(mesh_H, W, H, spd, nVerT=nVerT, op=op)
                    M = adap.metricIntersection(mesh_H, W, M, M2) if op.mtype == 'b' else M2
            if op.gradate:
                M_ = adap.isotropicMetric(W, inte.interp(mesh_H, H0)[0], bdy=True, op=op) # Initial boundary metric
                M = adap.metricIntersection(mesh_H, W, M, M_, bdy=True)
                adap.metricGradation(mesh_H, M)
                # TODO: always gradate to coast
            if op.advect:
                M = adap.advectMetric(M, u, 2*Dt, n=3*rm)
                # TODO: isotropic advection?

            # Adapt mesh and interpolate variables
            mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
            V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
            q_ = inte.mixedPairInterp(mesh_H, V_H, q_)[0]
            b = inte.interp(mesh_H, b)[0]
            q = Function(V_H)
            u, eta = q.split()
            u.rename("uv_2d")
            eta.rename("elev_2d")

            # Re-establish variational form
            qt = TestFunction(V_H)
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

            if outputOF:
                iA = form.indicator(V_H.sub(1), x1=490e3, x2=640e3, y1=4160e3, y2=4360e3, smooth=True)

        # Solve problem at current timestep
        adaptSolver.solve()
        q_.assign(q)

        # Estimate OF using trapezium rule TODO: allow for t-adaptivity
        if outputOF:
            step = assemble(eta * iA * dx)
            if (t >= op.Tstart) and not started:
                started = True
                J_trap = step
            elif t >= op.Tend:
                J_trap += step
            elif started:
                J_trap += 2 * step

        if cnt % ndump == 0:
            adaptiveFile.write(u, eta, time=t)
            if op.gauges:
                gaugeData = tim.extractTimeseries(gauges, eta, t, gaugeData, v0, op=op)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)
    J_h = J_trap * dt
    J = 2.4391e+13      # Objective functional value converged to 3s.f.
    print('J_h = %5.4e' % J_h)
    print('Relative error = %5.4e' % (np.abs(J - J_h) / J))

# Print to screen timing analyses
if getData and useAdjoint:
    msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)

# Save and plot timeseries
name = input("Enter a name for these time series (e.g. 'goalBased8-12-17'): ") or 'test'
for gauge in gauges:
    tim.saveTimeseries(gauge, gaugeData, name=name)