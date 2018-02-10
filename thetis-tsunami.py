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

def solverSW(startRes, approach, getData=True, getError=True, useAdjoint=True, mode='tohoku',
             op=opt.Options()):
    """
    Run mesh adaptive simulations for the Tohoku problem.

    :param startRes: Starting resolution, if bootstrapping is not used.
    :param approach: meshing method.
    :param getData: run forward simulation?
    :param getError: generate error estimates?
    :param useAdjoint: run adjoint simulation?
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

    # Load Mesh(es)
    if mode == 'tohoku':
        nEle = op.meshes[startRes]
        mesh_H, eta0, b = msh.TohokuDomain(nEle)  # Computational mesh
    elif mode == 'shallow-water':
        lx = 2 * np.pi
        n = pow(2, startRes)
        mesh_H = SquareMesh(n, n, lx, lx)  # Computational mesh
        nEle = msh.meshStats(mesh_H)[0]
        x, y = SpatialCoordinate(mesh_H)
    else:
        raise NotImplementedError

    if mode == 'shallow-water':
        P1_2d = FunctionSpace(mesh_H, "CG", 1)
        eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        b = Function(P1_2d).assign(0.1)

    # Define initial FunctionSpace and variables of problem and apply initial conditions
    V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
    q = Function(V_H)
    uv_2d, elev_2d = q.split()
    elev_2d.interpolate(eta0, annotate=False)
    uv_2d.rename("uv_2d")
    elev_2d.rename("elev_2d")

    # Establish finer mesh (h < H) upon which to approximate error
    if approach in ('explicit', 'goalBased'):
        mesh_h = adap.isoP2(mesh_H)
        V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)
        b_h = msh.TohokuDomain(mesh=mesh_h)[2]

    # Specify physical and solver parameters
    if mode == 'tohoku':
        dt = adap.adaptTimestepSW(mesh_H, b)
    else:
        dt = 0.1  # TODO: change this
    msc.dis('Using initial timestep = %4.3fs\n' % dt, op.printStats)
    Dt = Constant(dt)
    T = op.Tend
    iStart = int(op.Tstart / dt)
    iEnd = int(np.ceil(T / dt))

    if op.orderChange or approach == 'implicit':
        V_oi = VectorFunctionSpace(mesh_H, op.space1, op.degree1 + op.orderChange) \
               * FunctionSpace(mesh_H, op.space2, op.degree2 + op.orderChange)
        q_oi = Function(V_oi)
        uv_2d_oi, elev_2d_oi = q_oi.split()
        q__oi = Function(V_oi)
        uv_2d_oi_, elev_2d_oi_ = q__oi.split()
        b_oi = Function(V_oi.sub(1)).interpolate(b)
        if useAdjoint:
            dual_oi = Function(V_oi)
            dual_oi_u, dual_oi_e = dual_oi.split()

    # Define Functions relating to goalBased approach
    if approach in ('explicit', 'goalBased'):
        rho = Function(V_oi if op.orderChange else V_h)
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
            uv_2d_h, elev_2d_h = qh.split()
            uv_2d_h.rename("Fine velocity")
            elev_2d_h.rename("Fine elevation")
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
    if approach == 'implicit':
        e_ = Function(V_oi)
        e_0, e_1 = e_.split()
        e_0.interpolate(Expression([0, 0]))
        e_1.interpolate(Expression(0))
        e = Function(V_oi, name="Implicit error estimate")
        et = TestFunction(V_oi)
        (et0, et1) = (as_vector((et[0], et[1])), et[2])
        normal = FacetNormal(mesh_H)
    if approach in ('explicit', 'fluxJump', 'implicit', 'adjointBased', 'goalBased'):
        if approach in ('adjointBased', 'fluxJump', 'implicit') or op.orderChange:
            P0 = FunctionSpace(mesh_H, "DG", 0)
        else:
            P0 = FunctionSpace(mesh_h, "DG", 0)
        v = TestFunction(P0)
        epsilon = Function(P0, name="Error indicator")

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

    if getData:
        msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
        primalTimer = clock()

        # TODO: use proper Thetis forms to calculate implicit error and residuals

        # Get solver parameter values and construct solver
        fixedSolver = solver2d.FlowSolver2d(mesh_H, b)
        fixedOpt = fixedSolver.options
        fixedOpt.element_family = op.family
        fixedOpt.use_nonlinear_equations = False
        fixedOpt.use_grad_depth_viscosity_term = False
        fixedOpt.simulation_export_time = dt * op.ndump      # This might differ across error estimates
        fixedOpt.simulation_end_time = T
        fixedOpt.timestepper_type = op.timestepper
        fixedOpt.timestep = dt
        fixedOpt.output_directory = dirName
        fixedOpt.export_diagnostics = True
        fixedOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

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
        fixedSolver.assign_initial_conditions(elev=eta0)
        # fixedSolver.iterate(export_func=includeIt)
        fixedSolver.iterate()
        primalTimer = clock() - primalTimer
        print('Time elapsed for fixed mesh solver: %.1fs (%.2fmins)' % (primalTimer, primalTimer / 60))

        primalTimer = clock() - primalTimer
        msc.dis('Primal run complete. Run time: %.3fs' % primalTimer, op.printStats)

        # Reset counters
        if approach in ('explicit', 'fluxJump', 'implicit'):
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
        for k in range(0, iEnd, op.rm):
            msc.dis('Generating error estimate %d / %d' % (k / op.rm + 1, iEnd / op.rm + 1), op.printStats)
            indexStr = msc.indexString(k)

            # TODO: Load forward data from HDF5
            # TODO: estimate OF using trapezium rule and output (inc. fixed mesh case)
            # TODO: approximate residuals / implicit error (this might be better done in initial forward solve)
            # TODO: load adjoint data from HDF5
            # TODO: form error estimates

        errorTimer = clock() - errorTimer
        msc.dis('Errors estimated. Run time: %.3fs' % errorTimer, op.printStats)

    if approach in ('hessianBased', 'explicit', 'fluxJump', 'implicit', 'adjointBased', 'goalBased'):

        # Reset initial conditions
        if approach != 'hessianBased':
            uv_2d.interpolate(Expression([0, 0]))
            elev_2d.interpolate(eta0)
        if op.gradate:
            H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
        msc.dis('\nStarting adaptive mesh primal run (forwards in time)', op.printStats)
        adaptTimer = clock()
        while cnt < np.ceil(T / dt):
            stepTimer = clock()

            # Load variables from disk
            if cnt != 0:
                indexStr = msc.indexString(int(cnt / op.ndump))
                with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            # Construct metric
            W = TensorFunctionSpace(mesh_H, "CG", 1)
            if approach in ('explicit', 'fluxJump', 'adjointBased', 'goalBased'):
                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                if op.mtype != 's':
                    if op.iso:
                        M = adap.isotropicMetric(W, elev_2d, op=op)
                    else:
                        H = adap.constructHessian(mesh_H, W, elev_2d, op=op)
                        M = adap.computeSteadyMetric(mesh_H, W, H, elev_2d, nVerT=nVerT, op=op)
                if op.mtype != 'f':
                    spd = Function(FunctionSpace(mesh_H, 'DG', 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                    if op.iso:
                        M2 = adap.isotropicMetric(W, spd, op=op)
                    else:
                        H = adap.constructHessian(mesh_H, W, spd, op=op)
                        M2 = adap.computeSteadyMetric(mesh_H, W, H, spd, nVerT=nVerT, op=op)
                    M = adap.metricIntersection(mesh_H, W, M, M2) if op.mtype == 'b' else M2
            if op.gradate:
                M_ = adap.isotropicMetric(W, inte.interp(mesh_H, H0)[0], bdy=True, op=op)  # Initial boundary metric
                M = adap.metricIntersection(mesh_H, W, M, M_, bdy=True)
                adap.metricGradation(mesh_H, M, iso=op.iso)

            # Adapt mesh and interpolate variables
            mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
            elev_2d, uv_2d, b = inte.interp(mesh_H, elev_2d, uv_2d, b)
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')

            # Get mesh stats
            nEle = msh.meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt*dt, dt)

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh_H, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = False
            adapOpt.use_grad_depth_viscosity_term = False
            adapOpt.simulation_export_time = dt * op.ndump
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
    approach, getData, getError, useAdjoint = msc.cheatCodes(input(
        "Choose error estimator: 'hessianBased', 'explicit', 'fluxJump', 'implicit', 'adjointBased' or 'goalBased': "))
    if mode == 'tohoku':
        op = opt.Options(vscale=0.1 if approach == 'goalBased' else 0.85,
                         family='dg-dg',
                         # timestepper='SSPRK33', # 3-stage, 3rd order Strong Stability Preserving Runge Kutta
                         rm=60 if useAdjoint else 30,
                         gradate=True if (useAdjoint or approach == 'explicit') else False,
                         advect=False,
                         window=True if approach == 'adjointBased' else False,
                         outputMetric=False,
                         plotpvd=False,
                         gauges=False,
                         tAdapt=False,
                         bootstrap=False,
                         printStats=False,
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
                         window=True if approach == 'adjointBased' else False,
                         vscale=0.4 if useAdjoint else 0.85,
                         orderChange=0,
                         plotpvd=False)
    else:
        raise NotImplementedError

    # Run simulation(s)
    minRes = 0 if mode == 'tohoku' else 1
    maxRes = 4 if mode == 'tohoku' else 6
    textfile = open('outdata/outputs/' + mode + '/' + approach + date + '.txt', 'w+')
    for i in range(minRes, maxRes + 1):
        av, rel, timing = solverSW(i, approach, getData, getError, useAdjoint, mode=mode, op=op)
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
