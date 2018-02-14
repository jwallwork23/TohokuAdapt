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
    dirName = 'plots/'+mode+'/'
    if op.plotpvd:
        residualFile = File(dirName + "residual.pvd")
        implicitErrorFile = File(dirName + "implicitError.pvd")
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

    # Define Functions relating to a posteriori estimators
    if aposteriori or approach == 'norm':
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
        else:
            mesh_h = adap.isoP2(mesh_H)
            V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)
            if mode == 'tohoku':
                b_h = msh.TohokuDomain(mesh=mesh_h)[2]
            qh = Function(V_h)
            uh, eh = qh.split()
            uh.rename("Fine velocity")
            eh.rename("Fine elevation")
            if useAdjoint:
                dual_h = Function(V_h)
                dual_h_u, dual_h_e = dual_h.split()
                dual_h_u.rename('Fine adjoint velocity')
                dual_h_e.rename('Fine adjoint elevation')
        if useAdjoint:
            dual = Function(V_H)
            dual_u, dual_e = dual.split()
            dual_u.rename("Adjoint velocity")
            dual_e.rename("Adjoint elevation")
            if mode == 'tohoku':
                J = form.objectiveFunctionalSW(q, plot=True)
            elif mode == 'shallow-water':
                J = form.objectiveFunctionalSW(q, Tstart=op.Tstart, x1=0., x2=0.5*np.pi, y1=0.5 * np.pi,y2=1.5 * np.pi,
                                               smooth=False)
        if approach in ('implicit', 'DWE'):
            e_ = Function(V_oi)
            e = Function(V_oi)
            e0, e1 = e.split()
            e0.rename("Implicit error 0")
            e1.rename("Implicit error 1")
            et = TestFunction(V_oi)
            (et0, et1) = (as_vector((et[0], et[1])), et[2])
            normal = FacetNormal(mesh_H)
        if approach in ('residual', 'explicit', 'DWR'):
            rho = Function(V_oi if op.orderChange else V_h)
            rho_u, rho_e = rho.split()
            rho_u.rename("Velocity residual")
            rho_e.rename("Elevation residual")
            P0 = FunctionSpace(mesh_H if op.orderChange else mesh_h, "DG", 0)
        else:
            P0 = FunctionSpace(mesh_H, "DG", 0)
        v = TestFunction(P0)
        epsilon = Function(P0, name="Error indicator")

    # Initialise adaptivity placeholders and counters
    mM = [nEle, nEle]                               # Min/max #Elements
    Sn = nEle
    nVerT = msh.meshStats(mesh_H)[1] * op.vscale    # Target #Vertices
    endT = 0.
    cnt = 0

    # Get timestep
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    solver_obj.create_equations()
    dt = min(np.abs(solver_obj.compute_time_step().dat.data)) if mode == 'tohoku' else 0.025
    Dt = Constant(dt)
    iEnd = int(T / (dt * op.rm))

    if getData:
        msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
        primalTimer = clock()

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

        # Output error data
        if approach == 'fixedMesh' and op.outputOF:
            cb = err.TohokuCallback(solver_obj) if mode == 'tohoku' else err.ShallowWaterCallback(solver_obj)
            cb.output_dir = dirName
            cb.append_to_log = True
            cb.export_to_hdf5 = False
            solver_obj.add_callback(cb, 'timestep')

        # Apply ICs and time integrate
        solver_obj.assign_initial_conditions(elev=eta0)
        if aposteriori and approach != 'DWF':       # TODO: can these not go somewhere else?
            if mode == 'tohoku':
                def selector():
                    t = solver_obj.simulation_time
                    rm = 30                         # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt

                    finished = True if t > solver_obj.options.simulation_end_time - 0.5 * dt else False
                    if t < 0.5 * dt:
                        adj_start_timestep()
                    else:
                        adj_inc_timestep(time=t, finished=finished)     # TODO: test this. Shouldn't we use at every timestep?

            else:
                def selector():
                    t = solver_obj.simulation_time
                    rm = 10                         # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt

                    finished = True if t > solver_obj.options.simulation_end_time - 0.5 * dt else False
                    if t < 0.5 * dt:
                        adj_start_timestep()
                    else:
                        adj_inc_timestep(time=t, finished=finished)
            solver_obj.iterate(export_func=selector)
        else:
            solver_obj.iterate()
        if op.outputOF:
            J_h = err.getOF(dirName)    # Evaluate objective functional
        primalTimer = clock() - primalTimer
        msc.dis('Primal run complete. Run time: %.3fs' % primalTimer, op.printStats)

        # Reset counters
        cntT = int(np.ceil(T/dt))
        cnt = 0 if aposteriori and not useAdjoint else cntT

        if useAdjoint:
            save = True
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
                    msc.dis('Adjoint simulation %.2f%% complete' % ((cntT - cnt) / cntT * 100), op.printStats)
                    cnt -= 1
                    save = False
                else:
                    save = True
                if cnt == -1:
                    break
            dualTimer = clock() - dualTimer
            msc.dis('Dual run complete. Run time: %.3fs' % dualTimer, op.printStats)
            cnt += 1

    # Loop back over times to generate error estimators
    if getError:
        msc.dis('\nStarting error estimate generation', op.printStats)
        errorTimer = clock()

        # Define implicit error problem     # TODO: use Thetis forms
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
                uv_2d_oi.interpolate(uv_2d)
                elev_2d_oi.interpolate(elev_2d)
                uv_2d_oi_.interpolate(uv_2d_)
                elev_2d_oi_.interpolate(elev_2d_)
                if approach in ('implicit', 'DWE'):
                    errorSolver.solve()
                    e_.assign(e)
                    if approach == 'implicit':
                        epsilon = assemble(v * sqrt(inner(e, e)) * dx)
                    if op.plotpvd:
                        implicitErrorFile.write(e0, e1, time=float(k))

            # Approximate residuals
            if approach in ('explicit', 'residual', 'DWR'):
                if op.orderChange:
                    Au, Ae = form.strongResidualSW(q_oi, q_oi_, b, Dt)
                else:
                    qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                    Au, Ae = form.strongResidualSW(qh, q_h, b_h if mode == 'tohoku' else b, Dt)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)   # TODO: No idea why this isn't working in refinement case.
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=float(k))
                if approach == 'residual':
                    epsilon = assemble(v * sqrt(inner(rho, rho)) * dx)
                elif approach == 'explicit':
                    epsilon = err.explicitErrorEstimator(q_oi if op.orderChange else q_h, rho,
                                                         b if (op.orderChange or mode != 'tohoku') else b_h, v,
                                                         maxBathy=True if mode == 'tohoku' else False)
            epsilon.dat.data[:] = np.abs(epsilon.dat.data) * nVerT / (np.abs(assemble(epsilon * dx)) or 1.)  # Normalise
            epsilon.rename("Error indicator")

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
        if aposteriori:     # Reset initial conditions
            uv_2d.interpolate(Expression([0, 0]))
            elev_2d.interpolate(eta0)
            epsilon = Function(P0, name="Error indicator")
        if op.gradate:      # Get initial boundary metric
            H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))
        msc.dis('\nStarting adaptive mesh primal run (forwards in time)', op.printStats)
        adaptTimer = clock()
        while cnt < np.ceil(T / dt):
            stepTimer = clock()

            # Load variables from disk
            if cnt != 0:
                V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
                q = Function(V_H)
                uv_2d, elev_2d = q.split()
                with DumbCheckpoint(dirName+'hdf5/Elevation2d_'+msc.indexString(int(cnt/op.ndump)), mode=FILE_READ) \
                        as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(dirName+'hdf5/Velocity2d_'+msc.indexString(int(cnt/op.ndump)), mode=FILE_READ) \
                        as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            # Construct metric
            W = TensorFunctionSpace(mesh_H, "CG", 1)
            if aposteriori:
                with DumbCheckpoint(dirName+'hdf5/'+approach+'Error'+msc.indexString(int(cnt/op.rm)), mode=FILE_READ) \
                        as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)
            else:
                if approach == 'norm':
                    v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    epsilon = assemble(v * inner(q, q) * dx)
                    M = adap.isotropicMetric(W, epsilon, invert=False, nVerT=nVerT, op=op)
                elif approach =='fluxJump' and cnt != 0:
                    v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    epsilon = err.fluxJumpError(q, v)
                    M = adap.isotropicMetric(W, epsilon, invert=False, nVerT=nVerT, op=op)
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
            if op.plotpvd:
                File('plots/'+mode+'/mesh.pvd').write(mesh_H.coordinates, time=float(cnt))

            # Adapt mesh and interpolate variables
            if not (((approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.mtype != 'f')
                     or approach == 'fluxJump') and cnt == 0):
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

            # Evaluate callbacks and iterate
            if op.outputOF:
                cb = err.TohokuCallback(adapSolver) if mode == 'tohoku' else err.ShallowWaterCallback(adapSolver)
                cb.output_dir = dirName
                cb.append_to_log = True
                cb.export_to_hdf5 = False
                if cnt != 0:
                    cb.objective_functional = J_h
                adapSolver.add_callback(cb, 'timestep')
            adapSolver.iterate()
            if op.outputOF:
                J_h = err.getOF(dirName)  # Evaluate objective functional

            # Get mesh stats
            nEle = msh.meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt*dt, dt)
            cnt += op.rm

        adaptTimer = clock() - adaptTimer
        msc.dis('Adaptive primal run complete. Run time: %.3fs' % adaptTimer, op.printStats)
    else:
        av = nEle

    # Print to screen timing analyses and plot timeseries
    if op.printStats:
        msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer)
    rel = np.abs(op.J(mode) - J_h)/np.abs(op.J(mode))

    return av, rel, clock() - tic


if __name__ == '__main__':

    # Choose mode and set parameter values
    mode = input("Choose problem: 'tohoku', 'shallow-water', 'rossby-wave': ")
    approach, getData, getError, useAdjoint, aposteriori = msc.cheatCodes(input(
"""Choose error estimator from {'norm', 'fieldBased', 'gradientBased', 'hessianBased', 
'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', 'DWR' or 'DWE'}: """))
    op = opt.Options(vscale=0.1 if approach == 'DWR' else 0.85,
                     family='dg-dg',
                     rm=60 if useAdjoint else 30,
                     gradate=True if (useAdjoint or approach in ('explicit', 'implicit')) else False,
                     advect=False,
                     window=True if approach == 'DWF' else False,
                     outputMetric=False,
                     plotpvd=True,
                     gauges=False,
                     tAdapt=False,
                     bootstrap=False,
                     printStats=True,
                     outputOF=True,
                     orderChange=1 if approach in ('explicit', 'DWR', 'residual') else 0,
                     # orderChange=0,
                     ndump=10)
    if mode == 'shallow-water':
        op.Tstart = 0.5
        op.Tend = 2.5
        op.hmin = 5e-2
        op.hmax = 1.
        op.rm = 20 if useAdjoint else 10
        op.ndump = 10

    # Run simulation(s)
    textfile = open('outdata/outputs/'+mode+'/'+approach+date+'.txt', 'w+')
    for i in range(0, 7):
        av, rel, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
        print('Run %d:  Mean element count %6d  Relative error %.4f     Timing %.1fs' % (i, av, rel, timing))
        textfile.write('%d, %.4f, %.1f\n' % (av, rel, timing))
    textfile.close()
