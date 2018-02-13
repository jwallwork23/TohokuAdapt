from thetis import *
from thetis.field_defs import field_metadata
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
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

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
        residualFile = File(dirName + "residual.pvd")
        errorFile = File(dirName + "errorIndicator.pvd")

    # Load Mesh, initial condition and bathymetry
    if mode == 'tohoku':
        nEle = op.meshes[startRes]
        mesh_H, eta0, b = msh.TohokuDomain(nEle)
    elif mode == 'shallow-water':
        lx = 2 * np.pi
        n = pow(2, startRes)
        mesh_H = SquareMesh(n, n, lx, lx)
        nEle = msh.meshStats(mesh_H)[0]
        x, y = SpatialCoordinate(mesh_H)
        P1_2d = FunctionSpace(mesh_H, "CG", 1)
        eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        b = Function(P1_2d).assign(0.1)
    else:
        raise NotImplementedError
    T = op.Tend

    # Define initial FunctionSpace and variables of problem and apply initial conditions
    V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
    q = Function(V_H)
    uv_2d, elev_2d = q.split()
    elev_2d.interpolate(eta0, annotate=False)
    uv_2d.rename("uv_2d")
    elev_2d.rename("elev_2d")
    if approach in ('residual', 'implicit', 'DWR', 'DWE', 'explicit'):
        q_ = Function(V_H)
        uv_2d_, elev_2d_ = q_.split()

    # Establish finer mesh (h < H) upon which to approximate error
    if not op.orderChange:
        mesh_h = adap.isoP2(mesh_H)
        V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)
        b_h = msh.TohokuDomain(mesh=mesh_h)[2]
        qh = Function(V_h)
        uh, eh = qh.split()
        uh.rename("Fine velocity")
        eh.rename("Fine elevation")

    if op.orderChange:
        V_oi = VectorFunctionSpace(mesh_H, op.space1, op.degree1 + op.orderChange) \
               * FunctionSpace(mesh_H, op.space2, op.degree2 + op.orderChange)
        q_oi = Function(V_oi)
        uv_2d_oi, elev_2d_oi = q_oi.split()
        q_oi_ = Function(V_oi)
        uv_2d_oi_, elev_2d_oi_ = q_oi_.split()
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
            P0 = FunctionSpace(mesh_H, "DG", 0) if op.orderChange else FunctionSpace(mesh_h, "DG", 0)
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
                J = form.objectiveFunctionalSW(q, Tstart=op.Tstart, x1=0., x2=np.pi / 2, y1=0.5 * np.pi,y2=1.5 * np.pi,
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
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    nVerT = msh.meshStats(mesh_H)[1] * op.vscale  # Target #Vertices
    endT = 0.
    cnt = 0
    save = True

    # Get timestep
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    if mode == 'tohoku':
        solver_obj.create_equations()
        dt = min(np.abs(solver_obj.compute_time_step().dat.data))
    else:
        dt = 0.025
    Dt = Constant(dt)
    iEnd = int(T / (dt * op.rm))

    if getData:
        msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
        primalTimer = clock()

        # TODO: use proper Thetis forms to calculate implicit error and residuals

        # Get solver parameter values and construct solver
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = False
        options.use_grad_depth_viscosity_term = False
        options.simulation_export_time = dt * (op.rm-1) if aposteriori else dt * op.ndump
        options.simulation_end_time = T
        options.timestepper_type = op.timestepper
        options.timestep = dt
        options.output_directory = dirName
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        # solver_obj.export_initial_state = False if aposteriori else True

        # TODO: Compute adjoint solutions THIS DOESN'T WORK
        # def includeIt():
        #     if useAdjoint:
        #         # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
        #         if cnt >= int(T/op.ndump):
        #             finished = True
        #         if cnt == 0:
        #             adj_start_timestep()
        #         else:
        #             adj_inc_timestep(time=cnt*dt, finished=finished)
        #     cnt += 1

        # Apply ICs and time integrate
        solver_obj.assign_initial_conditions(elev=eta0)

        if aposteriori and approach != 'DWF':
            if mode == 'tohoku':
                def selector():
                    t = solver_obj.simulation_time
                    rm = 30                          # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    if int(t / dt) % rm == 0:
                        options.simulation_export_time = dt
                    else:
                        options.simulation_export_time = (rm - 1) * dt
            else:
                def selector():
                    t = solver_obj.simulation_time
                    rm = 10                          # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    if int(t / dt) % rm == 0:
                        options.simulation_export_time = dt
                    else:
                        options.simulation_export_time = (rm - 1) * dt
            solver_obj.iterate(export_func=selector)
        else:
            solver_obj.iterate()

        primalTimer = clock() - primalTimer
        print('Time elapsed for fixed mesh solver: %.1fs (%.2fmins)' % (primalTimer, primalTimer / 60))

        primalTimer = clock() - primalTimer
        msc.dis('Primal run complete. Run time: %.3fs' % primalTimer, op.printStats)

        # Reset counters
        if aposteriori and not useAdjoint:
            cnt = 0
        else:
            cntT = cnt = np.ceil(T / dt)

        if useAdjoint:
            parameters["adjoint"]["stop_annotating"] = True  # Stop registering equations
            msc.dis('\nStarting fixed mesh dual run (backwards in time)', op.printStats)
            dualTimer = clock()
            for (variable, solution) in compute_adjoint(J):
                if save:
                    # Load adjoint data and save to HDF5
                    dual.assign(variable, annotate=False)
                    dual_u, dual_e = dual.split()
                    dual_u.rename('Adjoint velocity')
                    dual_e.rename('Adjoint elevation')
                    with DumbCheckpoint(dirName + 'hdf5/adjoint_' + msc.indexString(cnt), mode=FILE_CREATE) as saveAdj:
                        saveAdj.store(dual_u)
                        saveAdj.store(dual_e)
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

        # Define implicit error problem
        if approach in ('implicit', 'DWE'):
            B_, L = form.formsSW(q_oi, q_oi_, et, b, Dt, allowNormalFlow=False)
            B = form.formsSW(e, e_, et, b, Dt, allowNormalFlow=False)[0]
            I = form.interelementTerm(et1 * uv_2d_oi, n=normal) * dS
            errorProblem = NonlinearVariationalProblem(B - L + B_ - I, e)
            errorSolver = NonlinearVariationalSolver(errorProblem, solver_parameters=op.params)

        for k in range(0, iEnd):
            msc.dis('Generating error estimate %d / %d' % (k+1, iEnd), op.printStats)

            if approach == 'DWF':
                with DumbCheckpoint(dirName+'hdf5/Velocity2d_'+msc.indexString(k), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(dirName+'hdf5/Elevation2d_'+msc.indexString(k), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
            else:
                i1 = 0 if k == 0 else 2*k
                i2 = 0 if k == 0 else 2*k+1
                with DumbCheckpoint(dirName+'hdf5/Velocity2d_'+msc.indexString(i1), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(dirName+'hdf5/Elevation2d_'+msc.indexString(i1), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
                uv_2d_.assign(uv_2d)
                elev_2d_.assign(elev_2d)
                with DumbCheckpoint(dirName+'hdf5/Velocity2d_'+msc.indexString(i2), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(dirName+'hdf5/Elevation2d_'+msc.indexString(i2), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()

            # Solve implicit error problem
            if approach in ('implicit', 'DWE') or op.orderChange:
                uv_2d_oi.interpolate(uv_2d, annotate=False)
                elev_2d_oi.interpolate(elev_2d, annotate=False)
                uv_2d_oi_.interpolate(uv_2d_, annotate=False)
                elev_2d_oi_.interpolate(elev_2d_, annotate=False)
                if approach in ('implicit', 'DWE'):
                    errorSolver.solve(annotate=False)
                    e_.assign(e)
                    if approach == 'implicit':
                        epsilon = assemble(v * sqrt(inner(e, e)) * dx)

            # Approximate residuals
            if approach in ('explicit', 'residual', 'DWR'):
                if op.orderChange:
                    Au, Ae = form.strongResidualSW(q_oi, q_oi_, b, Dt)
                else:
                    qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                    if mode == 'tohoku':
                        Au, Ae = form.strongResidualSW(qh, q_h, b_h, Dt)
                    else:
                        Au, Ae = form.strongResidualSW(qh, q_h, b, Dt)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=float(k))
                if approach == 'residual':
                    epsilon = assemble(v * sqrt(inner(rho, rho)) * dx)

            epsilon.rename("Error indicator")

            # TODO: Estimate OF using trapezium rule and output (inc. fixed mesh case)
            # TODO: Approximate residuals
            # TODO: Load adjoint data from HDF5
            # TODO: Form remaining error estimates

            # Store error estimates
            with DumbCheckpoint(dirName+'hdf5/'+approach+'Error'+msc.indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        msc.dis('Errors estimated. Run time: %.3fs' % errorTimer, op.printStats)

    if approach != 'fixedMesh':

        # Reset initial conditions
        if aposteriori:
            uv_2d.interpolate(Expression([0, 0]))
            elev_2d.interpolate(eta0)
            epsilon = Function(P0, name="Error indicator")
        if op.gradate:
            H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
        msc.dis('\nStarting adaptive mesh primal run (forwards in time)', op.printStats)
        adaptTimer = clock()
        while cnt < np.ceil(T / dt):
            stepTimer = clock()
            indexStr = msc.indexString(int(cnt / op.rm))

            # Load variables from disk
            if cnt != 0:
                with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            # Construct metric
            W = TensorFunctionSpace(mesh_H, "CG", 1)
            if aposteriori:
                with DumbCheckpoint(dirName+'hdf5/'+approach+'Error'+indexStr, mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
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
                            M = adap.isotropicMetric(W, elev_2d, invert=False, nVerT=nVerT, op=op)
                        elif approach == 'gradientBased':
                            g = adap.constructGradient(mesh_H, elev_2d)
                            M = adap.isotropicMetric(W, g, invert=False, nVerT=nVerT, op=op)
                        elif approach == 'hessianBased':
                            H = adap.constructHessian(mesh_H, W, elev_2d, op=op)
                            M = adap.computeSteadyMetric(mesh_H, W, H, elev_2d, nVerT=nVerT, op=op)
                    if cnt != 0:    # Can't adapt to zero velocity
                        if op.mtype != 'f':
                            spd = Function(FunctionSpace(mesh_H, 'DG', 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
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
                M_ = adap.isotropicMetric(W, inte.interp(mesh_H, H0)[0], bdy=True, op=op)  # Initial boundary metric
                M = adap.metricIntersection(mesh_H, W, M, M_, bdy=True)
                adap.metricGradation(mesh_H, M, iso=op.iso)

            # Adapt mesh and interpolate variables
            if not (approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.mtype != 'f' and cnt == 0):
                mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                elev_2d, uv_2d, b = inte.interp(mesh_H, elev_2d, uv_2d, b)
                uv_2d.rename('uv_2d')
                elev_2d.rename('elev_2d')

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh_H, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = False
            adapOpt.use_grad_depth_viscosity_term = False
            adapOpt.simulation_export_time = dt * op.ndump  # TODO: in this run could save every `rm`?
            startT = endT
            endT += dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = dt
            adapOpt.output_directory = dirName
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
            e = exporter.ExportManager(dirName + 'hdf5',
                                       ['elev_2d', 'uv_2d'],
                                       field_dict,
                                       field_metadata,
                                       export_type='hdf5')
            adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
            adapSolver.i_export = int(cnt / op.ndump)
            adapSolver.iteration = cnt
            adapSolver.simulation_time = startT
            adapSolver.next_export_t = startT + adapOpt.simulation_export_time  # For next export
            for e in adapSolver.exporters.values():
                e.set_next_export_ix(adapSolver.i_export)
            adapSolver.iterate()

            # Get mesh stats
            nEle = msh.meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt*dt, dt)
            cnt += op.rm

            # TODO: Estimate OF using trapezium rule, using a DiagnosticCallback object

        adaptTimer = clock() - adaptTimer
        # msc.dis('Adaptive primal run complete. Run time: %.3fs \nRelative error = %5.4f' % (adaptTimer, rel),
        #         op.printStats)
    else:
        av = nEle

    # Print to screen timing analyses and plot timeseries
    if op.printStats:
        msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer)
    rel = 0.        # TODO: compute this

    return av, rel, clock() - tic


if __name__ == '__main__':

    # Choose mode and set parameter values
    mode = input("Choose problem: 'tohoku', 'shallow-water', 'rossby-wave': ")
    approach, getData, getError, useAdjoint, aposteriori = msc.cheatCodes(input(
"""Choose error estimator from {'norm', 'fieldBased', 'gradientBased', 'hessianBased', 
'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', 'DWR' or 'DWE'}: """))
    if mode == 'tohoku':
        op = opt.Options(vscale=0.1 if approach == 'DWR' else 0.85,
                         family='dg-dg',
                         # timestepper='SSPRK33', # 3-stage, 3rd order Strong Stability Preserving Runge Kutta
                         rm=60 if useAdjoint else 30,
                         gradate=True if (useAdjoint or approach == 'explicit') else False,
                         advect=False,
                         window=True if approach == 'DWF' else False,
                         outputMetric=False,
                         plotpvd=False,
                         gauges=False,
                         tAdapt=False,
                         bootstrap=False,
                         printStats=False,
                         outputOF=True,
                         orderChange=1 if approach in ('explicit', 'DWR', 'residual') else 0,
                         ndump=10,
                         # iso=False if approach == 'hessianBased' else True,       # TODO: fix isotropic gradation
                         iso=False)
    elif mode == 'shallow-water':
        op = opt.Options(Tstart=0.5,
                         Tend=2.5,
                         hmin=5e-2,
                         hmax=1.,
                         rm=10,
                         ndump=5,
                         gradate=False,
                         bootstrap=False,
                         printStats=True,
                         outputOF=True,
                         advect=False,
                         window=True if approach == 'DWF' else False,
                         vscale=0.4 if useAdjoint else 0.85,
                         orderChange=1 if approach in ('explicit', 'DWR', 'residual') else 0,
                         plotpvd=True)
    else:
        raise NotImplementedError

    # Run simulation(s)
    minRes = 0 if mode == 'tohoku' else 1
    maxRes = 4 if mode == 'tohoku' else 6
    textfile = open('outdata/outputs/' + mode + '/' + approach + date + '.txt', 'w+')
    for i in range(minRes, maxRes + 1):
        av, rel, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
        exit(23)
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
