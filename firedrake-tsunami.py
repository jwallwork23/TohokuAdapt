from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock


import numpy as np
from time import clock
import datetime

import utils.adaptivity as adap
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt
import utils.timeseries as tim


now = datetime.datetime.now()
date = str(now.day)+'-'+str(now.month)+'-'+str(now.year%2000)


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
    try:
        assert(mode in ('tohoku', 'shallow-water', 'rossby-wave'))
    except:
        raise NotImplementedError
    primalTimer = dualTimer = errorTimer = adaptTimer = False
    if approach in ('implicit', 'DWE'):
        op.orderChange = 1

    # Establish filenames
    di = 'plots/' + mode + '/'
    if op.plotpvd:
        forwardFile = File(di + "forward.pvd")
        residualFile = File(di + "residual.pvd")
        errorFile = File(di + "errorIndicator.pvd")
    adaptiveFile = File(di + approach + ".pvd")
    if op.outputMetric:
        metricFile = File(di + "metric.pvd")
    if useAdjoint:
        adjointFile = File(di + "adjoint.pvd")

    # Load Mesh(es)
    if mode == 'tohoku':
        mesh_H, eta0, b = msh.TohokuDomain(startRes)    # Computational mesh
        dt = adap.adaptTimestepSW(mesh_H, b)
    elif mode == 'shallow-water':
        lx = 2 * np.pi
        n = pow(2, startRes)
        mesh_H = SquareMesh(n, n, lx, lx)  # Computational mesh
        x, y = SpatialCoordinate(mesh_H)
        P1 = FunctionSpace(mesh_H, "CG", 1)
        eta0 = Function(P1).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), annotate=False)
        b = Function(P1).assign(0.1, annotate=False)
        dt = 0.01
    elif mode == 'rossby-wave':
        n = 5
        lx = 48
        ly = 24
        mesh_H = RectangleMesh(lx * n, ly * n, lx, ly)
        xy = Function(mesh_H.coordinates)
        xy.dat.data[:, :] -= [lx / 2, ly / 2]
        mesh_H.coordinates.assign(xy)
        P1 = FunctionSpace(mesh_H, "CG", 1)
        b = Function(P1).assign(1.)
        dt = 0.1
    else:
        raise NotImplementedError

    # Define initial FunctionSpace and variables of problem and apply initial conditions
    V_H = op.mixedSpace(mesh_H)
    if mode == 'rossby-wave':
        q0 = form.solutionHuang(V_H, t=0.)
        u0, eta0 = q0.split()
        bc = DirichletBC(V_H.sub(0), [0, 0], 'on_boundary')
        f = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
    else:
        bc = []
        f = None
    q_ = Function(V_H)
    u_, eta_ = q_.split()
    eta_.interpolate(eta0)
    q = Function(V_H)
    q.assign(q_)
    u, eta = q.split()
    u.rename("uv_2d")
    eta.rename("elev_2d")

    # Establish finer mesh (h < H) upon which to approximate error
    if op.orderChange == 0:
        mesh_h = adap.isoP2(mesh_H)
        V_h = op.mixedSpace(mesh_h)
        if mode == 'tohoku':
            b_h = msh.TohokuDomain(mesh=mesh_h)[2]
        qh = Function(V_h)
        uh, eh = qh.split()
        uh.rename("Fine velocity")
        eh.rename("Fine elevation")

    # Specify physical and solver parameters
    if op.printStats:
        print('Using initial timestep = %4.3fs\n' % dt)
    Dt = Constant(dt)
    T = op.Tend
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
        V_oi = op.mixedSpace(mesh_H, orderChange=op.orderChange)
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
            if useAdjoint and op.orderChange == 0:
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
        if approach in ('implicit', 'DWE'):
            e_ = Function(V_oi)
            e = Function(V_oi)
            e0, e1 = e.split()
            e0.rename("Implicit error 0")
            e1.rename("Implicit error 1")

    # Initialise adaptivity placeholders and counters
    nEle, nVerT = msh.meshStats(mesh_H)
    nVerT *= op.vscale                              # Target #Vertices
    mM = [nEle, nEle]                               # Min/max #Elements
    Sn = nEle
    t = 0.
    cnt = 0
    # save = True

    # Calculate OF value
    switch = Constant(0.)
    k = Function(V_H)
    k0, k1 = k.split()
    k1.assign(form.indicator(V_H.sub(1), mode=mode))
    if useAdjoint:
        Jfunc = assemble(switch * inner(k, q_) * dx)
        Jfuncs = [Jfunc]
    # Jval = assemble(switch * inner(k, q_) * dx)
    # Jvals = [Jval]

    if getData:
        # Define variational problem
        # TODO: use LinearVariationalProblem, for which need TrialFunction(s) and a = lhs(F) and L = RHS(F)
        forwardProblem = NonlinearVariationalProblem(
            form.weakResidualSW(q, q_, b, Dt, coriolisFreq=f, impermeable=True, op=op), q, bcs=bc)
        forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

        if approach in ('implicit', 'DWE'):
            et = TestFunction(V_oi)
            (et0, et1) = (as_vector((et[0], et[1])), et[2])
            normal = FacetNormal(mesh_H)
            B_, L = form.formsSW(q_oi, q_oi_, b, Dt, coriolisFreq=f, impermeable=True)
            B = form.formsSW(e, e_, b, Dt, impermeable=True)[0]
            I = form.interelementTerm(et1 * u_oi, n=normal) * dS
            errorProblem = NonlinearVariationalProblem(B - L + B_ - I, e)
            errorSolver = NonlinearVariationalSolver(errorProblem, solver_parameters=op.params)

        if op.printStats:
            print('Starting fixed mesh primal run (forwards in time)')
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
                e_.assign(e, annotate=False)

            # Approximate residual of forward equation and save to HDF5
            if cnt % op.rm == 0:
                indexStr = msc.indexString(cnt)
                if approach == 'implicit':
                    epsilon = assemble(v * sqrt(inner(e, e)) * dx)
                    with DumbCheckpoint(di + 'hdf5/error_' + indexStr, mode=FILE_CREATE) as saveErr:
                        saveErr.store(epsilon)
                        saveErr.close()
                elif approach == 'DWE':
                    with DumbCheckpoint(di + 'hdf5/implicitError_' + indexStr, mode=FILE_CREATE) as saveIE:
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
                        with DumbCheckpoint(di + 'hdf5/residual_' + indexStr, mode=FILE_CREATE) as saveRes:
                            saveRes.store(rho_u)
                            saveRes.store(rho_e)
                            saveRes.close()
                        if op.plotpvd:
                            residualFile.write(rho_u, rho_e, time=t)
                if approach in ('DWF', 'explicit', 'fluxJump'):
                    with DumbCheckpoint(di + 'hdf5/forward_' + indexStr, mode=FILE_CREATE) as saveFor:
                        saveFor.store(u)
                        saveFor.store(eta)
                        saveFor.close()

            # Update solution at previous timestep
            q_.assign(q)
            if t >= op.Tstart:
                switch.assign(1.)
            if useAdjoint:
                Jfunc = assemble(switch * inner(k, q_) * dx)
                Jfuncs.append(Jfunc)
            # if approach == 'fixedMesh':
            #     Jval = assemble(switch * inner(k, q_) * dx)
            #     Jvals.append(Jval)

            # # Mark timesteps to be used in adjoint simulation     TODO: Update to pyadjoint
            # if useAdjoint:
            #     if t == 0.:
            #         adj_start_timestep()
            #     else:
            #         adj_inc_timestep(time=t, finished=t >= T)

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
        if op.printStats:
            print('Primal run complete. Run time: %.3fs' % primalTimer)

        # Establish OF
        if useAdjoint:
            J = assemble(Constant(0.) * inner(k, q_) * dx)
            for i in range(1, len(Jfuncs)):
                J += 0.5 * (Jfuncs[i - 1] + Jfuncs[i]) * dt
        # if op.outputOF and approach == 'fixedMesh':
        #     J_h = 0
        #     for i in range(1, len(Jvals)):
        #         J_h += 0.5 * (Jvals[i - 1] + Jvals[i]) * dt
        #     print('Estimated objective value J_h = ', J_h)

        # Reset counter in explicit case
        if aposteriori and not useAdjoint:
            cnt = 0

        if useAdjoint:
            # parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
            if op.printStats:
                print('\nStarting fixed mesh dual run (backwards in time)')
            dualTimer = clock()
            # for (variable, solution) in compute_adjoint(J):
            #     if save:
            #         # Load adjoint data and save to HDF5
            #         indexStr = msc.indexString(cnt)
            #
            #         dual.assign(variable, annotate=False)
            #         if op.window or ((op.orderChange or approach == 'DWF') and cnt % op.rm == 0):
            #             dual_u, dual_e = dual.split()
            #             dual_u.rename('Adjoint velocity')
            #             dual_e.rename('Adjoint elevation')
            #             with DumbCheckpoint(di + 'hdf5/adjoint_H_' + indexStr, mode=FILE_CREATE) as saveAdjH:
            #                 saveAdjH.store(dual_u)
            #                 saveAdjH.store(dual_e)
            #                 saveAdjH.close()
            #         elif (approach == 'DWR') and cnt % op.rm == 0:
            #             dual_h = inte.mixedPairInterp(mesh_h, V_h, dual)[0]
            #             dual_h_u, dual_h_e = dual_h.split()
            #             dual_h_u.rename('Fine adjoint velocity')
            #             dual_h_e.rename('Fine adjoint elevation')
            #             with DumbCheckpoint(di + 'hdf5/adjoint_' + indexStr, mode=FILE_CREATE) as saveAdj:
            #                 saveAdj.store(dual_h_u)
            #                 saveAdj.store(dual_h_e)
            #                 saveAdj.close()
            #             if op.printStats:
            #                 print('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT * 100))
            #         cnt -= 1
            #         save = False
            #     else:
            #         save = True
            #     if cnt == -1:
            #         break
            dJdb = compute_gradient(J, Control(b))
            tape = get_working_tape()
            solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]

            for i in range(len(solve_blocks) - 1, -1, -1):
                dual.assign(solve_blocks[i].adj_sol)        # TODO: in error estimation, can just extract later.
                if cnt % op.rm == 0:
                    adjointFile.write(dual_u, dual_e, time=t)
                    print('t = %.2fs' % t)
                t -= dt

            dualTimer = clock() - dualTimer
            if op.printStats:
                print('Adjoint run complete. Run time: %.3fs' % dualTimer)
            cnt += 1

    # Loop back over times to generate error estimators
    if getError:
        if op.printStats:
            print('\nStarting error estimate generation')
        errorTimer = clock()
        for k in range(0, iEnd, op.rm):
            if op.printStats:
                print('Generating error estimate %d / %d' % (k / op.rm + 1, iEnd / op.rm + 1))
            indexStr = msc.indexString(k)

            # Load forward / adjoint / residual data from HDF5
            if useAdjoint:
                if (approach == 'DWR') and not op.orderChange:
                    with DumbCheckpoint(di + 'hdf5/adjoint_' + indexStr, mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_h_u)
                        loadAdj.load(dual_h_e)
                        loadAdj.close()
                else:
                    with DumbCheckpoint(di + 'hdf5/adjoint_H_' + indexStr, mode=FILE_READ) as loadAdjH:
                        loadAdjH.load(dual_u)
                        loadAdjH.load(dual_e)
                        loadAdjH.close()
                    if op.orderChange:
                        dual_oi_u.interpolate(dual_u)
                        dual_oi_e.interpolate(dual_e)
            if (approach in ('explicit', 'DWF')):
                with DumbCheckpoint(di + 'hdf5/forward_' + indexStr, mode=FILE_READ) as loadFor:
                    loadFor.load(u)
                    loadFor.load(eta)
                    loadFor.close()
                if op.orderChange:
                    u_oi.interpolate(u)
                    eta_oi.interpolate(eta)
            if (approach in ('explicit', 'DWR')):
                with DumbCheckpoint(di + 'hdf5/residual_' + indexStr, mode=FILE_READ) as loadRes:
                    loadRes.load(rho_u)
                    loadRes.load(rho_e)
                    loadRes.close()
            elif approach == 'DWE':
                with DumbCheckpoint(di + 'hdf5/implicitError_' + indexStr, mode=FILE_READ) as loadIE:
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
                    with DumbCheckpoint(di + 'hdf5/adjoint_H_' + msc.indexString(i), mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_u)
                        loadAdj.load(dual_e)
                        loadAdj.close()
                    epsilon_ = err.basicErrorEstimator(q, dual, v)
                    for j in range(len(epsilon.dat.data)):
                        epsilon.dat.data[j] = max(epsilon.dat.data[j], epsilon_.dat.data[j])
            epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)  # Normalise
            epsilon.rename("Error indicator")

            # Store error estimates
            with DumbCheckpoint(di + 'hdf5/error_' + indexStr, mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        if op.printStats:
            print('Errors estimated. Run time: %.3fs' % errorTimer)

    if approach != 'fixedMesh':
        # Reset initial conditions
        if aposteriori:
            t = 0.
            switch.assign(0.)
            u_.interpolate(Expression([0, 0]))
            eta_.interpolate(eta0)
        if op.gradate:
            H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
        if op.printStats:
            print('\nStarting adaptive mesh primal run (forwards in time)')
        adaptTimer = clock()
        while t <= T:
            if cnt % op.rm == 0:
                stepTimer = clock()

                # Construct metric
                if aposteriori:
                    # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                    with DumbCheckpoint(di + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                        loadError.load(epsilon)
                        loadError.close()
                    errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                    M = adap.isotropicMetric(errEst, op=op, invert=False)
                else:
                    if approach in ('norm', 'fluxJump'):
                        v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                        norm = assemble(v * inner(q, q) * dx) if approach == 'norm' else err.fluxJumpError(q, v)
                        M = adap.isotropicMetric(norm, invert=False, nVerT=nVerT, op=op)
                    else:
                        if op.mtype != 's':
                            if approach == 'fieldBased':
                                M = adap.isotropicMetric(eta, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'gradientBased':
                                g = adap.constructGradient(eta)
                                M = adap.isotropicMetric(g, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'hessianBased':
                                M = adap.steadyMetric(eta, nVerT=nVerT, op=op)
                        if cnt != 0:    # Can't adapt to zero velocity
                            if op.mtype != 'f':
                                spd = Function(FunctionSpace(mesh_H, 'DG', 1)).interpolate(sqrt(dot(u, u)))
                                if approach == 'fieldBased':
                                    M2 = adap.isotropicMetric(spd, invert=False, nVerT=nVerT, op=op)
                                elif approach == 'gradientBased':
                                    g = adap.constructGradient(spd)
                                    M2 = adap.isotropicMetric(g, invert=False, nVerT=nVerT, op=op)
                                elif approach == 'hessianBased':
                                    M2 = adap.steadyMetric(spd, nVerT=nVerT, op=op)
                                M = adap.metricIntersection(M, M2) if op.mtype == 'b' else M2
                if op.gradate:
                    M_ = adap.isotropicMetric(inte.interp(mesh_H, H0)[0], bdy=True, op=op) # Initial boundary metric
                    M = adap.metricIntersection(M, M_, bdy=True)
                    adap.metricGradation(M, op=op)                  # TODO: always gradate to coast
                if op.advect:
                    M = adap.advectMetric(M, u, 2*Dt, n=3*op.rm)    # TODO: isotropic advection?
                if op.outputMetric:
                    M.rename("Metric")
                    metricFile.write(M, time=t)

                # Adapt mesh and interpolate variables
                if not (approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.mtype != 'f' and cnt == 0):
                    mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                    V_H = op.mixedSpace(mesh_H)
                    q_ = inte.mixedPairInterp(mesh_H, V_H, q_)[0]
                    if mode == 'tohoku':
                        b = inte.interp(mesh_H, b)[0]
                    elif mode == 'shallow-water':
                        b = Function(FunctionSpace(mesh_H, "CG", 1)).assign(0.1)
                    elif mode == 'rossby-wave':
                        P1 = FunctionSpace(mesh_H, "CG", 1)
                        b = Function(P1).assign(1.)
                        f = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
                    q = Function(V_H)
                    u, eta = q.split()
                    u.rename("uv_2d")
                    eta.rename("elev_2d")

                # Re-establish variational form
                adaptProblem = NonlinearVariationalProblem(
                    form.weakResidualSW(q, q_, b, Dt, coriolisFreq=f, impermeable=True, op=op), q)
                adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)
                if op.tAdapt:
                    dt = adap.adaptTimestepSW(mesh_H, b)
                    Dt.assign(dt)

                # Get mesh stats
                nEle = msh.meshStats(mesh_H)[0]
                mM = [min(nEle, mM[0]), max(nEle, mM[1])]
                Sn += nEle
                av = op.printToScreen(cnt/op.rm+1, clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, t, dt)

            # Solve problem at current timestep
            adaptSolver.solve()
            q_.assign(q)

            # # Calculate OF value
            # if t >= op.Tstart:
            #     switch.assign(1.)
            # Jval = assemble(switch * inner(k, q_) * dx)
            # Jvals.append(Jval)

            if cnt % op.ndump == 0:
                adaptiveFile.write(u, eta, time=t)
                if op.gauges:
                    gaugeData = tim.extractTimeseries(gauges, eta, t, gaugeData, v0, op=op)
                if op.printStats:
                    print('t = %.2fs' % t)
            t += dt
            cnt += 1
        adaptTimer = clock() - adaptTimer

        # Establish OF
        J_h = 0
        # for i in range(1, len(Jvals)):
        #     J_h += 0.5 * (Jvals[i - 1] + Jvals[i]) * dt
        if op.outputOF:
            print('Estimated objective value J_h = ', J_h)
        rel = np.abs((op.J(mode) - J_h) / op.J(mode))
        if op.printStats:
            print('Adaptive primal run complete. Run time: %.3fs \nRelative error = %5.4f' % (adaptTimer, rel))
    else:
        av = nEle

    # Print to screen timing analyses and plot timeseries
    if op.printStats:
        msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)
    if op.gauges:
        name = approach+date
        for gauge in gauges:
            tim.saveTimeseries(gauge, gaugeData, name=name)
    if not op.outputOF:
        rel = None

    return av, rel, clock()-tic


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}.")
    parser.add_argument("approach", help=
    """Choose error estimator from {'norm', 'fieldBased', 'gradientBased', 'hessianBased', 
    'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', 'DWR', 'DWE'}: """)
    args = parser.parse_args()
    print("Mode: ", args.mode)
    print("Approach: ", args.approach)
    mode = args.mode

    approach, getData, getError, useAdjoint, aposteriori = msc.cheatCodes(args.approach)
    op = opt.Options(vscale=0.1 if approach == 'DWR' else 0.85,
                     family='cg-cg' if mode == 'rossby-wave' else 'dg-dg',
                     rm=60 if useAdjoint else 30,
                     gradate=True if useAdjoint else False,
                     advect=False,
                     window=True if approach == 'DWF' else False,
                     outputMetric=False,
                     plotpvd=True,
                     gauges=False,
                     # iso=False if approach in ('gradientBased', 'hessianBased') else True,
                     iso=False,
                     printStats=True,
                     outputOF=True,
                     orderChange=1 if approach in ('explicit', 'DWR', 'residual') else 0,
                     # orderChange=0,
                     wd=False,
                     # wd=True if mode == 'tohoku' else False,
                     ndump=10)
    if mode == 'shallow-water':
        op.Tstart = 0.5
        op.Tend = 2.5
        op.hmin = 5e-2
        op.hmax = 1.
        op.rm = 20 if useAdjoint else 10
        op.ndump = 10

    # Run simulation(s)
    minRes = 0 if mode == 'tohoku' else 1
    maxRes = 4 if mode == 'tohoku' else 6
    textfile = open('outdata/outputs/'+mode+'/'+approach+date+'.txt', 'w+')
    for i in range(minRes, maxRes+1):
        av, rel, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
        print('Run %d:  Mean element count %6d  Relative error %.4f     Timing %.1fs' % (i, av, rel, timing))
        textfile.write('%d, %.4f, %.1f\n' % (av, rel, timing))
    textfile.close()
