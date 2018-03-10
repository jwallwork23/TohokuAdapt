from thetis import *
from thetis.field_defs import field_metadata
# from firedrake_adjoint import *

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

now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

def solverSW(startRes, approach, getData, getError, useAdjoint, aposteriori, mode='tohoku', op=opt.Options()):
    """
    Run mesh adaptive simulations for the Tohoku problem.

    :param startRes: Starting resolution.
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
        assert (float(physical_constants['g_grav'].dat.data) == 9.81)
    elif mode == 'shallow-water':
        msc.dis('*********************** SHALLOW WATER TEST PROBLEM ********************\n', op.printStats)
        assert (float(physical_constants['g_grav'].dat.data) == 9.81)
    elif mode == 'rossby-wave':
        msc.dis('****************** EQUATORIAL ROSSBY WAVE TEST PROBLEM ****************\n', op.printStats)
        assert (float(physical_constants['g_grav'].dat.data) == 1.)
    else:
        raise NotImplementedError
    primalTimer = dualTimer = errorTimer = adaptTimer = False
    if approach in ('implicit', 'DWE'):
        op.orderChange = 1

    # Establish filenames
    dirName = 'plots/'+mode+'/'
    if op.plotpvd:
        residualFile = File(dirName + "residual.pvd")
        implicitErrorFile = File(dirName + "implicitError.pvd")
        errorFile = File(dirName + "errorIndicator.pvd")

    # Load Mesh, initial condition and bathymetry
    if mode == 'tohoku':
        mesh_H, eta0, b = msh.TohokuDomain(startRes, wd=op.wd)
        P1 = FunctionSpace(mesh_H, "CG", 1)
    elif mode == 'shallow-water':
        lx = 2 * np.pi
        n = pow(2, startRes)
        mesh_H = SquareMesh(n, n, lx, lx)
        x, y = SpatialCoordinate(mesh_H)
        P1 = FunctionSpace(mesh_H, "CG", 1)
        eta0 = Function(P1).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        b = Function(P1).assign(0.1)
    elif mode == 'rossby-wave':
        n = pow(2, startRes)
        lx = 48
        ly = 24
        mesh_H = RectangleMesh(3 * lx * n, ly * n, 3 * lx, ly)
        xy = Function(mesh_H.coordinates)
        xy.dat.data[:, :] -= [3 * lx / 2, ly / 2]
        mesh_H.coordinates.assign(xy)
        P1 = FunctionSpace(mesh_H, "CG", 1)
        b = Function(P1).assign(1.)
    T = op.Tend

    # Define initial FunctionSpace and variables of problem and apply initial conditions
    V_H = VectorFunctionSpace(mesh_H, op.space1, op.degree1) * FunctionSpace(mesh_H, op.space2, op.degree2)
    if mode == 'rossby-wave':
        q = form.solutionHuang(V_H, t=0.)
        uv_2d, elev_2d = q.split()
        BCs = {1: {'uv': Constant(0.)}, 2: {'uv': Constant(0.)}, 3: {'uv': Constant(0.)}, 4: {'uv': Constant(0.)}}
    else:
        q = Function(V_H)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        BCs = {}
    uv_2d.rename("uv_2d")       # TODO: this should be 'Velocity2d'
    elev_2d.rename("elev_2d")
    if approach in ('residual', 'implicit', 'DWR', 'DWE', 'explicit'):
        q_ = Function(V_H)
        uv_2d_, elev_2d_ = q_.split()

    # Define Functions relating to a posteriori estimators
    P0 = FunctionSpace(mesh_H, "DG", 0)
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
            elif mode == 'shallow-water':
                b_h = Function(FunctionSpace(mesh_h, "CG", 1)).assign(0.1)
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
                J = form.objectiveFunctionalSW(q, plot=True)    # TODO: this no longer exists in pyadjoint
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
            if not op.orderChange:
                P0 = FunctionSpace(mesh_h, "DG", 0)
        epsilon = Function(P0, name="Error indicator")
    v = TestFunction(P0)

    if mode == 'rossby-wave':
        mesh_h = adap.isoP2(mesh_H)
        V_h = VectorFunctionSpace(mesh_h, op.space1, op.degree1) * FunctionSpace(mesh_h, op.space2, op.degree2)

    # Initialise adaptivity placeholders and counters
    nEle, nVerT = msh.meshStats(mesh_H)
    nVerT *= op.vscale                      # Target #Vertices
    mM = [nEle, nEle]                       # Min/max #Elements
    Sn = nEle
    endT = 0.

    # Get timestep
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    if mode == 'tohoku':
        solver_obj.create_equations()
        dt = min(np.abs(solver_obj.compute_time_step().dat.data))
    elif mode == 'shallow-water':
        dt = 0.025  # TODO: change this
    else:
        dt = 0.1    # TODO: change this
    Dt = Constant(dt)
    iEnd = int(np.ceil(T / (dt * op.rm)))
    if op.gradate or op.wd:                 # Get initial boundary metric
        H0 = Function(P1).interpolate(CellSize(mesh_H))
    if op.wd:
        g = adap.constructGradient(elev_2d)
        spd = assemble(v * sqrt(inner(g, g)) * dx)
        gs = np.min(np.abs(spd.dat.data))
        print('#### gradient = ', gs)
        ls = np.min([H0.dat.data[i] for i in DirichletBC(P1, 0, 'on_boundary').nodes])
        print('#### ls = ', ls)
        alpha = Constant(gs * ls)       # TODO: how to set wetting-and-drying parameter?
        print('#### alpha = ', alpha.dat.data)
        # alpha = Constant(0.5)
        # exit(23)

    if getData:
        msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
        primalTimer = clock()

        # Get solver parameter values and construct solver
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True if op.wd or mode == 'rossby-wave' else False
        options.use_grad_depth_viscosity_term = False
        if mode == 'rossby-wave':
            options.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
        options.simulation_export_time = dt * (op.rm-1) if aposteriori else dt * op.ndump
        options.simulation_end_time = T
        options.timestepper_type = op.timestepper
        options.timestep = dt
        options.output_directory = dirName
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        options.use_wetting_and_drying = op.wd
        if op.wd:
            options.wetting_and_drying_alpha = alpha

        # Output error data
        if approach == 'fixedMesh' and op.outputOF:
            if mode == 'tohoku':
                cb = err.TohokuCallback(solver_obj)
            elif mode == 'shallow-water':
                solver_obj.create_equations()
                cb = err.ShallowWaterCallback(solver_obj)
            elif mode == 'rossby-wave':
                solver_obj.create_equations()
                cb = err.RossbyWaveCallback(solver_obj)
            cb.output_dir = dirName
            cb.append_to_log = True
            cb.export_to_hdf5 = False
            solver_obj.add_callback(cb, 'timestep')

        # Apply ICs and time integrate
        if mode == 'rossby-wave':
            solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
        else:
            solver_obj.assign_initial_conditions(elev=eta0)
        solver_obj.bnd_functions['shallow_water'] = BCs
        if aposteriori and approach != 'DWF':
            if mode == 'tohoku':
                def selector():
                    t = solver_obj.simulation_time
                    rm = 30                         # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            elif mode == 'shallow-water':
                def selector():
                    t = solver_obj.simulation_time
                    rm = 10                         # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            else:
                def selector():
                    t = solver_obj.simulation_time
                    rm = 12                         # TODO: what can we do about this? Needs changing for adjoint
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
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
    cnt = 0

    # Loop back over times to generate error estimators
    if getError:
        msc.dis('\nStarting error estimate generation', op.printStats)
        errorTimer = clock()

        # Define implicit error problem     # TODO: use Thetis forms. Also put these in utils.forms
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
                i1 = 2*k
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
                    Au, Ae = form.strongResidualSW(q_oi, q_oi_, b, Dt, op=op)
                else:
                    qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                    Au, Ae = form.strongResidualSW(qh, q_h, b_h, Dt, op=op)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=float(k))
                if approach == 'residual':
                    epsilon = assemble(v * sqrt(inner(rho, rho)) * dx)
                elif approach == 'explicit':
                    epsilon = err.explicitErrorEstimator(q_oi if op.orderChange else q_h, rho,
                                                         b if (op.orderChange or mode != 'tohoku') else b_h, v,
                                                         maxBathy=True if mode == 'tohoku' else False)

            # TODO: Load adjoint data from HDF5
            # TODO: Form remaining error estimates
            # TODO: maximise DWF over time window

            # Store error estimates
            epsilon.rename("Error indicator")
            with DumbCheckpoint(dirName+'hdf5/'+approach+'Error'+msc.indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        msc.dis('Errors estimated. Run time: %.3fs' % errorTimer, op.printStats)

    if approach != 'fixedMesh':
        if aposteriori:     # Reset initial conditions
            if mode == 'rossby-wave':
                q = form.solutionHuang(V_H, t=0.)
                uv_2d, elev_2d = q.split()
            else:
                uv_2d.interpolate(Expression([0, 0]))
                elev_2d.interpolate(eta0)
            epsilon = Function(P0, name="Error indicator")
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
            if aposteriori:
                with DumbCheckpoint(dirName+'hdf5/'+approach+'Error'+msc.indexString(int(cnt/op.rm)), mode=FILE_READ) \
                        as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(inte.interp(mesh_H, epsilon)[0])
                M = adap.isotropicMetric(errEst, op=op, invert=False, nVerT=nVerT)
            else:
                if approach == 'norm':
                    v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    epsilon = assemble(v * inner(q, q) * dx)
                    M = adap.isotropicMetric(epsilon, invert=False, nVerT=nVerT, op=op)
                elif approach =='fluxJump' and cnt != 0:
                    v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                    epsilon = err.fluxJumpError(q, v)
                    M = adap.isotropicMetric(epsilon, invert=False, nVerT=nVerT, op=op)
                else:
                    if op.mtype != 's':
                        if approach == 'fieldBased':
                            M = adap.isotropicMetric(elev_2d, invert=False, nVerT=nVerT, op=op)
                        elif approach == 'gradientBased':
                            g = adap.constructGradient(elev_2d)
                            M = adap.isotropicMetric(g, invert=False, nVerT=nVerT, op=op)
                        elif approach == 'hessianBased':
                            M = adap.computeSteadyMetric(elev_2d, nVerT=nVerT, op=op)
                    if cnt != 0:    # Can't adapt to zero velocity
                        if op.mtype != 'f':
                            spd = Function(FunctionSpace(mesh_H, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                            if approach == 'fieldBased':
                                M2 = adap.isotropicMetric(spd, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'gradientBased':
                                g = adap.constructGradient(spd)
                                M2 = adap.isotropicMetric(g, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'hessianBased':
                                M2 = adap.computeSteadyMetric(spd, nVerT=nVerT, op=op)
                            M = adap.metricIntersection(M, M2) if op.mtype == 'b' else M2
            if op.gradate:
                M_ = adap.isotropicMetric(inte.interp(mesh_H, H0)[0], bdy=True, op=op)  # Initial boundary metric
                M = adap.metricIntersection(M, M_, bdy=True)
                adap.metricGradation(M, op=op)
            if op.plotpvd:
                File('plots/'+mode+'/mesh.pvd').write(mesh_H.coordinates, time=float(cnt))

            # Adapt mesh and interpolate variables
            if not (((approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.mtype != 'f')
                     or approach == 'fluxJump') and cnt == 0):
                mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                P1 = FunctionSpace(mesh_H, "CG", 1)
                elev_2d, uv_2d = inte.interp(mesh_H, elev_2d, uv_2d)
                if mode == 'tohoku':
                    b = inte.interp(mesh_H, b)[0]
                elif mode == 'shallow-water':
                    b = Function(P1).assign(0.1)
                else:
                    b = Function(P1).assign(1.)
                uv_2d.rename('uv_2d')
                elev_2d.rename('elev_2d')

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh_H, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True if op.wd or mode == 'rossby-wave' else False
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
            adapOpt.use_wetting_and_drying = op.wd
            if op.wd:
                adapOpt.wetting_and_drying_alpha = alpha
            if mode == 'rossby-wave':
                adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
            field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
            e = exporter.ExportManager(dirName + 'hdf5',
                                       ['elev_2d', 'uv_2d'],
                                       field_dict,
                                       field_metadata,
                                       export_type='hdf5')
            adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)    # TODO: why is this here, not later?
            adapSolver.i_export = int(cnt / op.ndump)
            adapSolver.iteration = cnt
            adapSolver.simulation_time = startT
            adapSolver.next_export_t = startT + adapOpt.simulation_export_time  # For next export
            for e in adapSolver.exporters.values():
                e.set_next_export_ix(adapSolver.i_export)

            # Evaluate callbacks and iterate
            if op.outputOF:
                if mode == 'tohoku':
                    cb = err.TohokuCallback(adapSolver)
                elif mode == 'shallow-water':
                    cb = err.ShallowWaterCallback(adapSolver)
                elif mode == 'rossby-wave':
                    cb = err.RossbyWaveCallback(adapSolver)
                cb.output_dir = dirName
                cb.append_to_log = True
                cb.export_to_hdf5 = False
                if cnt != 0:
                    cb.objective_functional = J_h
                adapSolver.add_callback(cb, 'timestep')
            solver_obj.bnd_functions['shallow_water'] = BCs
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

    if mode == 'rossby-wave':
        q_a = Function(V_h)
        u_a, eta_a = q_a.split()
        with DumbCheckpoint(dirName + 'hdf5/finalAnalytic', mode=FILE_READ) as loadAna:
            loadAna.load(u_a)
            loadAna.load(eta_a)
            loadAna.close()
        uv_2d, elev_2d = inte.mixedPairInterp(mesh_h, uv_2d, elev_2d)



    # Print to screen timing analyses and plot timeseries
    if op.printStats:
        msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)
    rel = np.abs(op.J(mode) - J_h)/np.abs(op.J(mode))

    return av, rel, J_h, clock() - tic


if __name__ == '__main__':

    # Choose mode and set parameter values
    mode = input("Choose problem: 'tohoku', 'shallow-water', 'rossby-wave': ")
    approach, getData, getError, useAdjoint, aposteriori = msc.cheatCodes(input(
"""Choose error estimator from {'norm', 'fieldBased', 'gradientBased', 'hessianBased', 
'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', 'DWR' or 'DWE'}: """))
    op = opt.Options(vscale=0.1 if approach in ('DWR', 'gradientBased') else 0.85,
                     family='dg-cg',
                     rm=60 if useAdjoint else 30,
                     gradate=True if aposteriori else False,
                     advect=False,
                     window=True if approach == 'DWF' else False,
                     outputMetric=False,
                     plotpvd=True,
                     gauges=False,
                     tAdapt=False,
                     # iso=False if approach in ('gradientBased', 'hessianBased') else True,
                     # iso=True,
                     iso=False,
                     bootstrap=False,
                     printStats=True,
                     outputOF=False,
                     orderChange=1 if approach in ('explicit', 'DWR', 'residual') else 0,
                     # orderChange=0,
                     wd=False,
                     # wd=True if mode == 'tohoku' else False,
                     ndump=10)
    if mode == 'shallow-water':
        op.Tstart = 0.5
        op.Tend = 2.5
        op.hmin = 1e-4
        op.hmax = 1.
        op.rm = 20 if useAdjoint else 10
        op.ndump = 10
    elif mode == 'rossby-wave':
        op.Tstart = 30.
        op.Tend = 120.
        op.hmin = 5e-3
        op.hmax = 10.
        op.rm = 24 if useAdjoint else 12
        op.ndump = 12

    # Run simulation(s)
    s = '_BOOTSTRAP' if op.bootstrap else ''
    textfile = open('outdata/outputs/'+mode+'/'+approach+date+s+'.txt', 'w+')
    if op.bootstrap:
        for i in range(11):
            av, rel, J_h, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
            var = np.abs(J_h - J_h_) if i > 0 else 0.
            J_h_ = J_h
            print('Run %d:  Mean element count %6d      Objective value %.4e        Timing %.1fs    Difference %.4e'
                  % (i, av, J_h, timing, var))
            textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, J_h, timing, var))
    else:
        # for i in range(6):
        for i in range(1, 6):
            av, rel, J_h, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
            print('Run %d:  Mean element count %6d      Relative error %.4e         Timing %.1fs'
                  % (i, av, rel, timing))
            textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, timing, J_h))
    textfile.close()
