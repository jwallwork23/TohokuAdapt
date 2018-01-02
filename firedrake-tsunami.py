from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
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
tAdapt = True

# Define initial mesh and mesh statistics placeholders
op = opt.Options(vscale=0.4 if useAdjoint else 0.85,
                 rm=60 if useAdjoint else 30,
                 gradate=True if useAdjoint else False,
                 advect=False,
                 window=False,
                 outputHessian=False,
                 plotpvd=False if approach == 'fixedMesh' else True,
                 coarseness=5,
                 gauges=True)

# TODO: bootstrap to establish initial mesh resolution

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
mesh, eta0, b = msh.TohokuDomain(op.coarseness)
if useAdjoint:
    try:
        assert op.coarseness != 1
    except:
        raise NotImplementedError("Requested mesh resolution not yet available.")
    mesh_N, b_N = msh.TohokuDomain(op.coarseness-1)[0::2]   # Get finer mesh and associated bathymetry
    V_N = VectorFunctionSpace(mesh_N, op.space1, op.degree1) * FunctionSpace(mesh_N, op.space2, op.degree2)

# Specify physical and solver parameters
h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
dt = 0.9 * min(h.dat.data) / np.sqrt(op.g * max(b.dat.data))
print('     #### Using initial timestep = %4.3fs\n' % dt)
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

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iStart = int(op.Tstart / dt)
iEnd = int(np.ceil(T / dt))
nEle, nVer = msh.meshStats(mesh)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices
nVerT0 = nVerT

# Initialise counters
t = 0.
cnt = 0

if getData:
    # Define variational problem
    qt = TestFunction(V)
    forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    print('\nStarting fixed mesh primal run (forwards in time)')
    finished = False
    primalTimer = clock()
    if op.plotpvd:
        forwardFile.write(u, eta, time=t)
    while t < T + dt:
        # Solve problem at current timestep
        forwardSolver.solve()

        # Approximate residual of forward equation and save to HDF5
        if useAdjoint:
            if not cnt % rm:
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

        if not cnt % ndump:
            if op.plotpvd:
                forwardFile.write(u, eta, time=t)
            if op.gauges and not useAdjoint:
                gaugeData = tim.extractTimeseries(gauges, eta, gaugeData, v0, op=op)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    cnt -=1
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    if useAdjoint:
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
                # Load adjoint data
                dual.assign(variable, annotate=False)
                dual_N = inte.mixedPairInterp(mesh_N, V_N, dual)[0]
                dual_N_u, dual_N_e = dual_N.split()
                dual_N_u.rename('Adjoint velocity')
                dual_N_e.rename('Adjoint elevation')

                # Save adjoint data to HDF5
                if not cnt % rm:
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_' + msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
                        saveAdj.store(dual_N_u)
                        saveAdj.store(dual_N_e)
                        saveAdj.close()

                if not cnt % ndump:
                    if op.plotpvd:
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

# Loop back over times to generate error estimators
if getError:
    errorTimer = clock()
    for k in range(0, iEnd, rm):
        print('Generating error estimate %d / %d' % (k / rm + 1, iEnd / rm))
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

        # Normalise error estimate
        # if k != 0:
        #     nVerT = 0.1 * nVerT0
        epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)
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
    if useAdjoint & op.gradate:
        h0 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    print('\nStarting adaptive mesh primal run (forwards in time)')
    adaptTimer = clock()
    while t <= T:
        if (cnt % rm == 0) & (np.abs(t-T) > 0.5 * dt):
            stepTimer = clock()

            # Construct metric
            W = TensorFunctionSpace(mesh, "CG", 1)
            if useAdjoint & (cnt != 0):
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                H = adap.constructHessian(mesh, W, eta, op=op)
                M = adap.computeSteadyMetric(mesh, W, H, eta, nVerT=nVerT, op=op)
            if op.gradate:
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
            adaptProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
            adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)

            # Get mesh stats
            nEle = msh.meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt/rm+1, clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, t)

        if tAdapt & (cnt % rm == 1):
            h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
            dt = 0.9 * min(h.dat.data) / np.sqrt(op.g * 0.1)
            Dt.assign(dt)
            print('     #### New timestep = %4.3fs' % dt)

        # Solve problem at current timestep
        adaptSolver.solve()
        q_.assign(q)

        if not cnt % ndump:
            if op.plotpvd:
                adaptiveFile.write(u, eta, time=t)
            if op.gauges:
                gaugeData = tim.extractTimeseries(gauges, eta, gaugeData, v0, op=op)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)
    cnt -= 1

# Print to screen timing analyses
if getData and useAdjoint:
    msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)

# Save and plot timeseries
name = input("Enter a name for these time series (e.g. 'goalBased8-12-17'): ") or 'test'
for gauge in gauges:
    tim.saveTimeseries(gauge, gaugeData[gauge], name=name)
    tim.plotGauges(gauge, int(T), op=op)
