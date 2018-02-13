from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock
import datetime

import utils.adaptivity as adap
import utils.bootstrapping as boot
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt
import utils.timeseries as tim


now = datetime.datetime.now()
date = str(now.day)+'-'+str(now.month)+'-'+str(now.year%2000)

# TODO: Homotopy method to consider a convex combination of error estimators?
# TODO: combine rossby-wave test case into this script
# TODO: consider dual weighted implicit error, as well as DWR. Perhaps a more generalised setting for error estimates


def solverSW(startRes, approach, getData, getError, useAdjoint, aposteriori, mode='tohoku', op=opt.Options()):
    """
    Run mesh adaptive simulations for the Tohoku problem.
    
    :param startRes: Starting resolution, if bootstrapping is not used.
    :param approach: meshing method.
    :param getData: run forward simulation?
    :param getError: generate error estimates?
    :param useAdjoint: run adjoint simulation?
    :param aposteriori: error estimator classification.
    :param mode: test case or main script used.
    :param op: parameter values.
    :return: mean element count and relative error in objective functional value.
    """
    tic = clock()
    if mode == 'tohoku':
        msc.dis('*********************** TOHOKU TSUNAMI SIMULATION *********************\n', op.printStats)
    elif mode == 'shallow-water':
        msc.dis('*********************** SHALLOW WATER TEST PROBLEM ********************\n', op.printStats)
    bootTimer = primalTimer = dualTimer = errorTimer = adaptTimer = False
    if approach in ('implicit', 'DWE'):
        op.orderChange = 1

    # Establish initial mesh resolution
    if op.bootstrap:
        bootTimer = clock()
        msc.dis('\nBootstrapping to establish optimal mesh resolution', op.printStats)
        startRes = boot.bootstrap(mode, tol=2e10 if mode == 'tohoku' else 1e-3)[0]
        bootTimer = clock() - bootTimer
        msc.dis('Bootstrapping run time: %.3fs\n' % bootTimer, op.printStats)

    # Establish filenames
    dirName = 'plots/' + mode + '/'
    if op.plotpvd:
        forwardFile = File(dirName + "forward.pvd")
        residualFile = File(dirName + "residual.pvd")
        errorFile = File(dirName + "errorIndicator.pvd")
    adaptiveFile = File(dirName + approach + ".pvd")
    if op.outputMetric:
        metricFile = File(dirName + "metric.pvd")

    # Load Mesh(es)
    if mode == 'tohoku':
        nEle = op.meshes[startRes]
        mesh_H, eta0, b = msh.TohokuDomain(nEle)    # Computational mesh
    elif mode == 'shallow-water':
        lx = 2 * np.pi
        n = pow(2, startRes)
        mesh_H = SquareMesh(n, n, lx, lx)  # Computational mesh
        nEle = msh.meshStats(mesh_H)[0]
        x, y = SpatialCoordinate(mesh_H)
        P1_2d = FunctionSpace(mesh_H, "CG", 1)
        eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        b = Function(P1_2d).assign(0.1)
    else:
        raise NotImplementedError

    # Define initial FunctionSpace and variables of problem and apply initial conditions
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

    # Establish finer mesh (h < H) upon which to approximate error
    if not op.orderChange:
        mesh_h = adap.isoP2(mesh_H)
        V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)
        if mode == 'tohoku':
            b_h = msh.TohokuDomain(mesh=mesh_h)[2]
        qh = Function(V_h)
        uh, eh = qh.split()
        uh.rename("Fine velocity")
        eh.rename("Fine elevation")

    # Specify physical and solver parameters
    dt = adap.adaptTimestepSW(mesh_H, b) if mode == 'tohoku' else 0.1        # TODO: change this
    msc.dis('Using initial timestep = %4.3fs\n' % dt, op.printStats)
    Dt = Constant(dt)
    T = op.Tend
    if op.tAdapt:
        # TODO: t-adaptive goal-based needs to acknowledge timestep change in initial run
        raise NotImplementedError("Mesh adaptive routines not quite calibrated for t-adaptivity")
    else:
        iStart = int(op.Tstart / dt)
        iEnd = int(np.ceil(T / dt))

    # Get initial gauge values
    if op.gauges:
        gaugeData = {}
        gauges = ("P02", "P06")
        v0 = {}
        for gauge in gauges:
            v0[gauge] = float(eta.at(op.gaugeCoord(gauge)))

    if op.orderChange:
        V_oi = VectorFunctionSpace(mesh_H, op.space1, op.degree1+op.orderChange) \
               * FunctionSpace(mesh_H, op.space2, op.degree2+op.orderChange)
        q_oi = Function(V_oi)
        u_oi, eta_oi = q_oi.split()
        q_oi_ = Function(V_oi)
        u_oi_, eta_oi_ = q_oi_.split()
        if useAdjoint:
            dual_oi = Function(V_oi)
            dual_oi_u, dual_oi_e = dual_oi.split()

    # Define Functions relating to a posteriori estimators
    if aposteriori or approach == 'norm':
        if approach in ('residual', 'explicit', 'DWR'):
            rho = Function(V_oi if op.orderChange else V_h)
            rho_u, rho_e = rho.split()
            rho_u.rename("Velocity residual")
            rho_e.rename("Elevation residual")
            if useAdjoint:
                dual_h = Function(V_h)
                dual_h_u, dual_h_e = dual_h.split()
                dual_h_u.rename('Fine adjoint velocity')
                dual_h_e.rename('Fine adjoint elevation')
            P0 = FunctionSpace(mesh_H if op.orderChange else mesh_h, "DG", 0)
        else:
            P0 = FunctionSpace(mesh_H, "DG", 0)
        v = TestFunction(P0)
        epsilon = Function(P0, name="Error indicator")
        if useAdjoint:
            dual = Function(V_H)
            dual_u, dual_e = dual.split()
            dual_u.rename("Adjoint velocity")
            dual_e.rename("Adjoint elevation")
            if mode == 'tohoku':
                J = form.objectiveFunctionalSW(q, plot=True)
            elif mode == 'shallow-water':
                J = form.objectiveFunctionalSW(q, Tstart=op.Tstart, x1=0., x2=np.pi / 2, y1=0.5 * np.pi, y2=1.5 * np.pi,
                                               smooth=False)
            else:
                raise NotImplementedError
        if approach in ('implicit', 'DWE'):
            e_ = Function(V_oi)
            e = Function(V_oi)
            e0, e1 = e.split()
            e0.rename("Implicit error 0")
            e1.rename("Implicit error 1")
            et = TestFunction(V_oi)
            (et0, et1) = (as_vector((et[0], et[1])), et[2])
            normal = FacetNormal(mesh_H)

    if op.outputOF:
        J_trap = 0.
        started = False

    # Initialise adaptivity placeholders and counters
    mM = [nEle, nEle]                               # Min/max #Elements
    Sn = nEle
    nVerT = msh.meshStats(mesh_H)[1] * op.vscale    # Target #Vertices
    t = 0.
    cnt = 0
    save = True

    if getData:
        # Define variational problem
        qt = TestFunction(V_H)
        forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
        forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

        if approach in ('implicit', 'DWE'):
            B_, L = form.formsSW(q_oi, q_oi_, et, b, Dt, allowNormalFlow=False)
            B = form.formsSW(e, e_, et, b, Dt, allowNormalFlow=False)[0]
            I = form.interelementTerm(et1 * u_oi, n=normal) * dS
            errorProblem = NonlinearVariationalProblem(B - L + B_ - I, e)
            errorSolver = NonlinearVariationalSolver(errorProblem, solver_parameters=op.params)

        if op.outputOF:
            if mode == 'tohoku':
                iA = form.indicator(V_H.sub(1), x1=490e3, x2=640e3, y1=4160e3, y2=4360e3, smooth=True)
            elif mode == 'shallow-water':
                iA = form.indicator(V_H.sub(1), x1=0., x2=0.5 * np.pi, y1=0.5 * np.pi, y2=1.5 * np.pi, smooth=False)

        msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
        finished = False
        primalTimer = clock()
        if op.plotpvd:
            forwardFile.write(u, eta, time=t)
        while t < T + dt:
            # Solve problem at current timestep
            forwardSolver.solve()

            if approach in ('implicit', 'DWE'):
                u_oi.interpolate(u, annotate=False)
                eta_oi.interpolate(eta, annotate=False)
                u_oi_.interpolate(u_, annotate=False)
                eta_oi_.interpolate(eta_, annotate=False)
                errorSolver.solve(annotate=False)
                e_.assign(e)

            # Approximate residual of forward equation and save to HDF5
            if cnt % op.rm == 0:
                indexStr = msc.indexString(cnt)
                if approach == 'implicit':
                    epsilon = assemble(v * sqrt(inner(e, e)) * dx)
                    with DumbCheckpoint(dirName + 'hdf5/error_' + indexStr, mode=FILE_CREATE) as saveErr:
                        saveErr.store(epsilon)
                        saveErr.close()
                elif approach == 'DWE':
                    with DumbCheckpoint(dirName + 'hdf5/implicitError_' + indexStr, mode=FILE_CREATE) as saveIE:
                        saveIE.store(e0)
                        saveIE.store(e1)
                        saveIE.close()
                elif approach in ('explicit', 'DWR'):
                    if op.orderChange:
                        u_oi.interpolate(u)
                        eta_oi.interpolate(eta)
                        u_oi_.interpolate(u_)
                        eta_oi_.interpolate(eta_)
                        Au, Ae = form.strongResidualSW(q_oi, q_oi_, b, Dt)
                    else:
                        qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                        if mode == 'tohoku':
                            Au, Ae = form.strongResidualSW(qh, q_h, b_h, Dt)
                        else:
                            Au, Ae = form.strongResidualSW(qh, q_h, b, Dt)
                    if approach in ('explicit', 'DWR'):
                        rho_u.interpolate(Au)
                        rho_e.interpolate(Ae)
                        with DumbCheckpoint(dirName + 'hdf5/residual_' + indexStr, mode=FILE_CREATE) as saveRes:
                            saveRes.store(rho_u)
                            saveRes.store(rho_e)
                            saveRes.close()
                        if op.plotpvd:
                            residualFile.write(rho_u, rho_e, time=t)
                if approach in ('DWF', 'explicit', 'fluxJump'):
                    with DumbCheckpoint(dirName + 'hdf5/forward_' + indexStr, mode=FILE_CREATE) as saveFor:
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

            # Estimate OF using trapezium rule TODO: allow for t-adaptivity
            if op.outputOF and approach == 'fixedMesh':
                step = assemble(eta * iA * dx)
                if (t >= op.Tstart) and not started:
                    started = True
                    J_trap = step
                elif t >= op.Tend:
                    J_trap += step
                elif started:
                    J_trap += 2 * step

            if cnt % op.ndump == 0:
                if op.plotpvd:
                    forwardFile.write(u, eta, time=t)
                if op.gauges and approach == 'fixedMesh':
                    gaugeData = tim.extractTimeseries(gauges, eta, t, gaugeData, v0, op=op)
                if op.printStats:
                    print('t = %.2fs' % t)
            t += dt
            cnt += 1
        cnt -=1
        cntT = cnt  # Total number of steps
        primalTimer = clock() - primalTimer
        if op.outputOF:
            rel = np.abs((op.J(mode) - J_trap * dt) / op.J(mode))
            # print('#### DEBUG: J_h = ', J_trap * dt)
        msc.dis('Primal run complete. Run time: %.3fs' % primalTimer, op.printStats)

        # Reset counter in explicit case
        if aposteriori and not useAdjoint:
            cnt = 0

        if useAdjoint:
            parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
            msc.dis('\nStarting fixed mesh dual run (backwards in time)', op.printStats)
            dualTimer = clock()
            for (variable, solution) in compute_adjoint(J):
                if save:
                    # Load adjoint data and save to HDF5
                    indexStr = msc.indexString(cnt)

                    dual.assign(variable, annotate=False)
                    if op.window or ((op.orderChange or approach == 'DWF') and cnt % op.rm == 0):
                        dual_u, dual_e = dual.split()
                        dual_u.rename('Adjoint velocity')
                        dual_e.rename('Adjoint elevation')
                        with DumbCheckpoint(dirName + 'hdf5/adjoint_H_' + indexStr, mode=FILE_CREATE) as saveAdjH:
                            saveAdjH.store(dual_u)
                            saveAdjH.store(dual_e)
                            saveAdjH.close()
                    elif (approach == 'DWR') and cnt % op.rm == 0:
                        dual_h = inte.mixedPairInterp(mesh_h, V_h, dual)[0]
                        dual_h_u, dual_h_e = dual_h.split()
                        dual_h_u.rename('Fine adjoint velocity')
                        dual_h_e.rename('Fine adjoint elevation')
                        with DumbCheckpoint(dirName + 'hdf5/adjoint_' + indexStr, mode=FILE_CREATE) as saveAdj:
                            saveAdj.store(dual_h_u)
                            saveAdj.store(dual_h_e)
                            saveAdj.close()
                        if op.printStats:
                            print('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT * 100))
                    cnt -= 1
                    save = False
                else:
                    save = True
                if cnt == -1:
                    break
            dualTimer = clock() - dualTimer
            msc.dis('Adjoint run complete. Run time: %.3fs' % dualTimer, op.printStats)
            cnt += 1

    # Loop back over times to generate error estimators
    if getError:
        msc.dis('\nStarting error estimate generation', op.printStats)
        errorTimer = clock()
        for k in range(0, iEnd, op.rm):
            msc.dis('Generating error estimate %d / %d' % (k / op.rm + 1, iEnd / op.rm + 1), op.printStats)
            indexStr = msc.indexString(k)

            # Load forward / adjoint / residual data from HDF5
            if useAdjoint:
                if (approach == 'DWR') and not op.orderChange:
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_' + indexStr, mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_h_u)
                        loadAdj.load(dual_h_e)
                        loadAdj.close()
                else:
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_H_' + indexStr, mode=FILE_READ) as loadAdjH:
                        loadAdjH.load(dual_u)
                        loadAdjH.load(dual_e)
                        loadAdjH.close()
                    if op.orderChange:
                        dual_oi_u.interpolate(dual_u)
                        dual_oi_e.interpolate(dual_e)
            if (approach in ('explicit', 'DWF')):
                with DumbCheckpoint(dirName + 'hdf5/forward_' + indexStr, mode=FILE_READ) as loadFor:
                    loadFor.load(u)
                    loadFor.load(eta)
                    loadFor.close()
                if op.orderChange:
                    u_oi.interpolate(u)
                    eta_oi.interpolate(eta)
            if (approach in ('explicit', 'DWR')):
                with DumbCheckpoint(dirName + 'hdf5/residual_' + indexStr, mode=FILE_READ) as loadRes:
                    loadRes.load(rho_u)
                    loadRes.load(rho_e)
                    loadRes.close()
            elif approach in ('implicit', 'DWE'):
                with DumbCheckpoint(dirName + 'hdf5/implicitError_' + indexStr, mode=FILE_READ) as loadIE:
                    loadIE.load(e0)
                    loadIE.load(e1)
                    loadIE.close()

            if approach == 'DWF':
                epsilon = err.basicErrorEstimator(q, dual, v)
            elif approach == 'DWR':
                epsilon = err.DWR(rho, dual_oi if op.orderChange else dual_h, v)
            elif approach == 'DWE':
                epsilon = err.DWR(e, dual_oi, v)
            elif approach == 'explicit':
                epsilon = err.explicitErrorEstimator(q_oi if op.orderChange else q, rho, b, v,
                                                     maxBathy=True if mode == 'tohoku' else False)

            # Loop over relevant time window
            if op.window:
                for i in range(k, min(k+iEnd-iStart, iEnd)):
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_H_' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_u)
                        loadAdj.load(dual_e)
                        loadAdj.close()
                    epsilon_ = err.basicErrorEstimator(q, dual, v)
                for j in range(len(epsilon.dat.data)):
                    epsilon.dat.data[j] = max(epsilon.dat.data[j], epsilon_.dat.data[j])
            epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)  # Normalise
            epsilon.rename("Error indicator")   # TODO: use L2 normalisation here ^^^ ?

            # Store error estimates
            with DumbCheckpoint(dirName + 'hdf5/error_' + indexStr, mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        msc.dis('Errors estimated. Run time: %.3fs' % errorTimer, op.printStats)

    if approach != 'fixedMesh':

        # Reset initial conditions
        if aposteriori:
            t = 0.
            u_.interpolate(Expression([0, 0]))
            eta_.interpolate(eta0)
        if op.gradate:
            H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
        msc.dis('\nStarting adaptive mesh primal run (forwards in time)', op.printStats)
        adaptTimer = clock()
        while t <= T:
            if cnt % op.rm == 0:      # TODO: change this condition for t-adaptivity?
                stepTimer = clock()

                # Construct metric
                W = TensorFunctionSpace(mesh_H, "CG", 1)
                if aposteriori:
                    # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                    with DumbCheckpoint(dirName + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                        loadError.load(epsilon)
                        loadError.close()
                    errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                    M = adap.isotropicMetric(W, errEst, op=op, invert=False)
                else:
                    if approach in ('norm', 'fluxJump'):
                        v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                        norm = assemble(v * inner(q, q) * dx) if approach == 'norm' else err.fluxJumpError(q, v)
                        M = adap.isotropicMetric(W, norm, invert=False, nVerT=nVerT, op=op)
                    else:
                        if op.mtype != 's':
                            if approach == 'fieldBased':
                                M = adap.isotropicMetric(W, eta, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'gradientBased':
                                g = adap.constructGradient(mesh_H, eta)
                                M = adap.isotropicMetric(W, g, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'hessianBased':
                                H = adap.constructHessian(mesh_H, W, eta, op=op)
                                M = adap.computeSteadyMetric(mesh_H, W, H, eta, nVerT=nVerT, op=op)
                        if cnt != 0:    # Can't adapt to zero velocity
                            if op.mtype != 'f':
                                spd = Function(FunctionSpace(mesh_H, 'DG', 1)).interpolate(sqrt(dot(u, u)))
                                if approach == 'fieldBased':
                                    M2 = adap.isotropicMetric(W, spd, invert=False, nVerT=nVerT, op=op)
                                elif approach == 'gradientBased':
                                    g = adap.constructGradient(mesh_H, spd)
                                    M2 = adap.isotropicMetric(W, g, invert=False, nVerT=nVerT, op=op)
                                elif approach == 'hessianBased':
                                    H = adap.constructHessian(mesh_H, W, spd, op=op)
                                    M2 = adap.computeSteadyMetric(mesh_H, W, H, spd, nVerT=nVerT, op=op)
                                M = adap.metricIntersection(mesh_H, W, M, M2) if op.mtype == 'b' else M2
                if op.gradate:
                    M_ = adap.isotropicMetric(W, inte.interp(mesh_H, H0)[0], bdy=True, op=op) # Initial boundary metric
                    M = adap.metricIntersection(mesh_H, W, M, M_, bdy=True)
                    adap.metricGradation(mesh_H, M, iso=op.iso)
                    # TODO: always gradate to coast
                if op.advect:
                    M = adap.advectMetric(M, u, 2*Dt, n=3*op.rm)
                    # TODO: isotropic advection?
                if op.outputMetric:
                    M.rename("Metric")
                    metricFile.write(M, time=t)

                # Adapt mesh and interpolate variables
                if not (approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.mtype != 'f' and cnt == 0):
                    mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                    V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
                    q_ = inte.mixedPairInterp(mesh_H, V_H, q_)[0]
                    if mode == 'tohoku':
                        b = inte.interp(mesh_H, b)[0]
                    q = Function(V_H)
                    u, eta = q.split()
                    u.rename("uv_2d")
                    eta.rename("elev_2d")

                # Re-establish variational form
                qt = TestFunction(V_H)
                adaptProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
                adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)
                if op.tAdapt:
                    dt = adap.adaptTimestepSW(mesh_H, b)
                    Dt.assign(dt)

                # Get mesh stats
                nEle = msh.meshStats(mesh_H)[0]
                mM = [min(nEle, mM[0]), max(nEle, mM[1])]
                Sn += nEle
                av = op.printToScreen(cnt/op.rm+1, clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, t, dt)
                if op.outputOF:
                    if mode == 'tohoku':
                        iA = form.indicator(V_H.sub(1), x1=490e3, x2=640e3, y1=4160e3, y2=4360e3, smooth=True)
                    elif mode == 'shallow-water':
                        iA = form.indicator(V_H.sub(1), x1=0., x2=0.5*np.pi, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)

            # Solve problem at current timestep
            adaptSolver.solve()
            q_.assign(q)

            # Estimate OF using trapezium rule TODO: allow for t-adaptivity
            if op.outputOF:
                step = assemble(eta * iA * dx)
                if (t >= op.Tstart) and not started:
                    started = True
                    J_trap = step
                elif t >= op.Tend:
                    J_trap += step
                elif started:
                    J_trap += 2 * step

            if cnt % op.ndump == 0:
                adaptiveFile.write(u, eta, time=t)
                if op.gauges:
                    gaugeData = tim.extractTimeseries(gauges, eta, t, gaugeData, v0, op=op)
                msc.dis('t = %.2fs' % t, op.printStats)
            t += dt
            cnt += 1
        adaptTimer = clock() - adaptTimer
        # print('#### DEBUG: J_h = ', J_trap * dt)
        rel = np.abs((op.J(mode) - J_trap * dt) / op.J(mode))
        msc.dis('Adaptive primal run complete. Run time: %.3fs \nRelative error = %5.4f' % (adaptTimer, rel), op.printStats)
    else:
        av = nEle

    # Print to screen timing analyses and plot timeseries
    if op.printStats:
        msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer)
    if op.gauges:
        name = approach+date
        for gauge in gauges:
            tim.saveTimeseries(gauge, gaugeData, name=name)
    if not op.outputOF:
        rel = None

    return av, rel, clock()-tic


if __name__ == '__main__':

    # Choose mode and set parameter values
    mode = input("Choose problem: 'tohoku', 'shallow-water', 'rossby-wave': ")
    approach, getData, getError, useAdjoint, aposteriori = msc.cheatCodes(input(
"""Choose error estimator from {'norm', 'fieldBased', 'gradientBased', 'hessianBased', 
'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', 'DWR' or 'DWE'}: """))
    if mode == 'tohoku':
        op = opt.Options(vscale=0.1 if approach == 'DWR' else 0.85,
                         # family='dg-dg',
                         rm=60 if useAdjoint else 30,
                         gradate=True if (useAdjoint or approach == 'explicit') else False,
                         advect=False,
                         window=True if approach == 'DWF' else False,
                         outputMetric=False,
                         plotpvd=False,
                         gauges=False,
                         tAdapt=False,
                         bootstrap=False,
                         printStats=True,
                         outputOF=True,
                         orderChange=0,
                         ndump=10,
                         # iso=False if approach == 'hessianBased' else True,       # TODO: fix isotropic gradation
                         iso=False)
    elif mode == 'shallow-water':
        op = opt.Options(Tstart=0.5,
                         Tend=2.5,
                         hmin=5e-2,
                         hmax=1.,
                         rm=5,
                         ndump=1,
                         gradate=False,
                         bootstrap=False,
                         printStats=False,
                         outputOF=True,
                         advect=False,
                         window=True if approach == 'DWF' else False,
                         vscale=0.4 if useAdjoint else 0.85,
                         orderChange=0,
                         plotpvd=False)
    else:
        raise NotImplementedError

    # Run simulation(s)
    minRes = 0 if mode == 'tohoku' else 1
    maxRes = 4 if mode == 'tohoku' else 6
    textfile = open('outdata/outputs/'+mode+'/'+approach+date+'.txt', 'w+')
    for i in range(minRes, maxRes+1):
        av, rel, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
        print('Run %d:  Mean element count %6d  Relative error %.4f     Timing %.1fs' % (i, av, rel, timing))
        textfile.write('%d, %.4f, %.1f\n' % (av, rel, timing))
        # try:
        #     av, rel, timing = solverSW(i, approach, getData, getError, useAdjoint, mode=mode, op=op)
        #     print('Run %d:  Mean element count %6d  Relative error %.4f     Timing %.1fs' % (i, av, rel, timing))
        #     textfile.write('%d , %.4f, %.1f\n' % (av, rel, timing))
        # except:
        #     print("#### ERROR: Failed to run simulation %d." % i)
    textfile.close()

    # TODO: loop over all mesh adaptive approaches consistently and then plot
