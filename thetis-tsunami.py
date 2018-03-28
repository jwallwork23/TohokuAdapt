from thetis import *
from thetis.field_defs import field_metadata
from firedrake_adjoint import *
import pyadjoint
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
    if mode == 'tohoku':
        msc.dis('*********************** TOHOKU TSUNAMI SIMULATION *********************\n', op.printStats)
        g = 9.81
    elif mode == 'shallow-water':
        msc.dis('*********************** SHALLOW WATER TEST PROBLEM ********************\n', op.printStats)
        g = 9.81
    elif mode == 'rossby-wave':
        msc.dis('****************** EQUATORIAL ROSSBY WAVE TEST PROBLEM ****************\n', op.printStats)
        g = 1.
    else:
        raise NotImplementedError
    try:
        assert (float(physical_constants['g_grav'].dat.data) == g)
    except:
        physical_constants['g_grav'].assign(g)
    primalTimer = dualTimer = errorTimer = adaptTimer = False
    if approach in ('implicit', 'DWE'):
        op.orderChange = 1

    # Establish filenames
    di = 'plots/'+mode+'/'
    if op.plotpvd:
        residualFile = File(di + "residual.pvd")
        implicitErrorFile = File(di + "implicitError.pvd")
        errorFile = File(di + "errorIndicator.pvd")
        adjointFile = File(di + "adjoint.pvd")

    # Load Mesh, initial condition and bathymetry
    if mode == 'tohoku':
        mesh_H, eta0, b, BCs = msh.TohokuDomain(startRes, wd=op.wd)
    elif mode == 'shallow-water':
        mesh_H, eta0, b, BCs = msh.domainSW(startRes)
    else:
        mesh_H, u0, eta0, b, BCs, f = msh.domainRW(startRes, op=op)
    T = op.Tend

    # Define initial FunctionSpace and variables of problem and apply initial conditions
    V_H = op.mixedSpace(mesh_H)
    q = Function(V_H)
    uv_2d, elev_2d = q.split()  # Needed to load data into
    uv_2d.rename("uv_2d")
    elev_2d.rename("elev_2d")
    P1 = FunctionSpace(mesh_H, "CG", 1)
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
        if approach in ('Implicit', 'DWE'):
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

    # Initialise adaptivity placeholders and counters
    nEle, nVerT = msh.meshStats(mesh_H)
    nVerT *= op.vscale                      # Target #Vertices
    mM = [nEle, nEle]                       # Min/max #Elements
    Sn = nEle
    endT = 0.

    # Get timestep
    solver_obj = solver2d.FlowSolver2d(mesh_H, b)
    if mode == 'shallow-water':
        dt = 0.025                          # TODO: change this
    else:
        # Ensure simulation time is achieved exactly.
        solver_obj.create_equations()
        dt = min(np.abs(solver_obj.compute_time_step().dat.data))
        for i in (3., 2.5, 2., 1.5, 1., 0.5, 0.25, 0.2, 0.1, 0.05):
            if dt > i:
                dt = i
                break
    Dt = Constant(dt)
    # iEnd = int(np.ceil(T / (dt * op.rm)))
    iEnd = int(T / (dt * op.rm))            # It appears this format is better for CFL criterion derived timesteps
    if op.gradate or op.wd:                 # Get initial boundary metric
        H0 = Function(P1).interpolate(CellSize(mesh_H))
    if op.wd:
        g = adap.constructGradient(elev_2d)
        spd = assemble(v * sqrt(inner(g, g)) * dx)
        gs = np.min(np.abs(spd.dat.data))
        print('#### gradient = ', gs)
        ls = np.min([H0.dat.data[i] for i in DirichletBC(P1, 0, 'on_boundary').nodes])
        print('#### ls = ', ls)
        alpha = Constant(gs * ls)           # TODO: How to set wetting-and-drying parameter?
        print('#### alpha = ', alpha.dat.data)
        # alpha = Constant(0.5)
        # exit(23)

    tic = clock()
    if getData:
        msc.dis('Starting fixed mesh primal run (forwards in time)', op.printStats)
        primalTimer = clock()

        # Get solver parameter values and construct solver
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True if op.nonlinear else False
        options.use_grad_depth_viscosity_term = False
        options.use_grad_div_viscosity_term = False
        if mode == 'rossby-wave':
            options.coriolis_frequency = f
        options.simulation_export_time = dt * (op.rm-1) if aposteriori else dt * op.ndump
        options.simulation_end_time = T
        options.timestepper_type = op.timestepper
        options.timestep = dt
        options.output_directory = di
        options.export_diagnostics = True
        options.log_output = op.printStats
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        options.use_wetting_and_drying = op.wd
        if op.wd:
            options.wetting_and_drying_alpha = alpha
        if mode == 'rossby-wave':
            solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        else:
            solver_obj.assign_initial_conditions(elev=eta0)
        if mode == 'tohoku':
            cb1 = err.TohokuCallback(solver_obj)
            cb2 = err.ObjectiveTohokuCallback(solver_obj)
        elif mode == 'shallow-water':
            cb1 = err.ShallowWaterCallback(solver_obj)
            cb2 = err.ObjectiveSWCallback(solver_obj)
        else:
            cb1 = err.RossbyWaveCallback(solver_obj)
            cb2 = err.ObjectiveRWCallback(solver_obj)
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.add_callback(cb2, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        if aposteriori and approach != 'DWF':
            if mode == 'tohoku':
                def selector():
                    t = solver_obj.simulation_time
                    # rm = 30                         # TODO: what can we do about this? Needs changing for adjoint
                    rm = 60
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            elif mode == 'shallow-water':
                def selector():
                    t = solver_obj.simulation_time
                    # rm = 10                         # TODO: what can we do about this? Needs changing for adjoint
                    rm = 20
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            else:
                def selector():
                    t = solver_obj.simulation_time
                    # rm = 24                         # TODO: what can we do about this? Needs changing for adjoint
                    rm = 48
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            solver_obj.iterate(export_func=selector)    # TODO: This ^^^ doesn't always work
        else:
            solver_obj.iterate()
        if op.outputOF:
            J_h = cb1.__call__()[1]    # Evaluate objective functional
        primalTimer = clock() - primalTimer
        msc.dis('Primal run complete. Run time: %.3fs' % primalTimer, op.printStats)

        # Reset counters
        cntT = int(np.ceil(T/dt))
        if useAdjoint:
            msc.dis('\nGenerating dual solutions...', op.printStats)
            dualTimer = clock()

            # Assemble objective functional
            Jfuncs = cb2.__call__()[1]
            J = 0
            for i in range(1, len(Jfuncs)):
                J += 0.5 * (Jfuncs[i - 1] + Jfuncs[i]) * dt

            # Compute gradient
            dJdb = compute_gradient(J, Control(b))      # TODO: Rewrite pyadjooint coode to avoid computing this
            File(di + 'gradient.pvd').write(dJdb)
            print("Norm of gradient = %e" % dJdb.dat.norm)

            # Extract adjoint solutions
            tape = get_working_tape()
            # tape.visualise()
            solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
            N = len(solve_blocks)
            r = N % op.rm   # Number of extra tape annotations in setup
            for i in range(N-1, r-2, -op.rm):
                dual.assign(solve_blocks[i].adj_sol)
                dual_u, dual_e = dual.split()
                dual_u.rename('Adjoint velocity')
                dual_e.rename('Adjoint elevation')
                with DumbCheckpoint(di+'hdf5/adjoint_'+msc.indexString(int((i-r+1)/op.rm)), mode=FILE_CREATE) as saveAdj:
                    saveAdj.store(dual_u)
                    saveAdj.store(dual_e)
                    saveAdj.close()
                if op.plotpvd:
                    adjointFile.write(dual_u, dual_e, time=dt * (i-r+1))
                if op.printStats:
                    print('Adjoint simulation %.2f%% complete' % ((N - i + r - 1) / N * 100))
            dualTimer = clock() - dualTimer
            msc.dis('Dual run complete. Run time: %.3fs' % dualTimer, op.printStats)
    cnt = 0

    # Loop back over times to generate error estimators
    if getError:
        msc.dis('\nStarting error estimate generation', op.printStats)
        errorTimer = clock()

        # Define implicit error problem
        if approach in ('implicit', 'DWE'): # TODO: Check out situation with nonlinear option
            B_, L = form.formsSW(q_oi, q_oi_, b, Dt, impermeable=False, coriolisFreq=f, nonlinear=True)
            B = form.formsSW(e, e_, b, Dt, impermeable=False, coriolisFreq=f, nonlinear=True)[0]
            I = form.interelementTerm(et1 * uv_2d_oi, n=normal) * dS
            errorProblem = NonlinearVariationalProblem(B - L + B_ - I, e)
            errorSolver = NonlinearVariationalSolver(errorProblem, solver_parameters=op.params)

        for k in range(0, iEnd):
            msc.dis('Generating error estimate %d / %d' % (k+1, iEnd), op.printStats)

            if approach == 'DWF':
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+msc.indexString(k), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+msc.indexString(k), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
            else:
                i1 = 0 if k == 0 else 2*k-1
                i2 = 2*k
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+msc.indexString(i1), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+msc.indexString(i1), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
                uv_2d_.assign(uv_2d)
                elev_2d_.assign(elev_2d)
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+msc.indexString(i2), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+msc.indexString(i2), mode=FILE_READ) as loadElev:
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
                    Au, Ae = form.strongResidualSW(q_oi, q_oi_, b, Dt, coriolisFreq=None, nonlinear=False, op=op)
                else:
                    qh, q_h = inte.mixedPairInterp(mesh_h, V_h, q, q_)
                    Au, Ae = form.strongResidualSW(qh, q_h, b_h, Dt, coriolisFreq=None, nonlinear=False, op=op)
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

            if useAdjoint:
                with DumbCheckpoint(di+'hdf5/adjoint_'+msc.indexString(k), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_u)
                    loadAdj.load(dual_e)
                    loadAdj.close()
                if approach == 'DWR':   # TODO: Also consider higher order / refined duals
                    epsilon = assemble(v * inner(rho, dual) * dx)
                elif approach == 'DWE':
                    epsilon = assemble(v* inner(e, dual) * dx)
                elif approach == 'DWF':
                    raise NotImplementedError
                    # TODO: maximise DWF over time window

            # Store error estimates
            epsilon.rename("Error indicator")
            with DumbCheckpoint(di+'hdf5/'+approach+'Error'+msc.indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        msc.dis('Errors estimated. Run time: %.3fs' % errorTimer, op.printStats)

    if approach != 'fixedMesh':
        with pyadjoint.stop_annotating():
            q = Function(V_H)
            uv_2d, elev_2d = q.split()
            elev_2d.interpolate(eta0)
            if aposteriori:
                epsilon = Function(P0, name="Error indicator")
            msc.dis('\nStarting adaptive mesh primal run (forwards in time)', op.printStats)
            adaptTimer = clock()
            # while cnt < np.ceil(T / dt):
            while cnt < int(T / dt):        # It appears this format is better for CFL criterion derived timesteps
                stepTimer = clock()

                # Load variables from disk
                if cnt != 0:
                    V_H = op.mixedSpace(mesh_H)
                    q = Function(V_H)
                    uv_2d, elev_2d = q.split()
                    with DumbCheckpoint(di+'hdf5/Elevation2d_'+msc.indexString(int(cnt/op.ndump)), mode=FILE_READ) \
                            as loadElev:
                        loadElev.load(elev_2d, name='elev_2d')
                        loadElev.close()
                    with DumbCheckpoint(di+'hdf5/Velocity2d_'+msc.indexString(int(cnt/op.ndump)), mode=FILE_READ) \
                            as loadVel:
                        loadVel.load(uv_2d, name='uv_2d')
                        loadVel.close()

                # Construct metric
                if aposteriori:
                    with DumbCheckpoint(di+'hdf5/'+approach+'Error'+msc.indexString(int(cnt/op.rm)), mode=FILE_READ) \
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
                adapOpt.use_nonlinear_equations = True if op.nonlinear else False
                adapOpt.use_grad_depth_viscosity_term = False
                adapOpt.use_grad_div_viscosity_term = False
                adapOpt.simulation_export_time = dt * op.ndump
                startT = endT
                endT += dt * op.rm
                adapOpt.simulation_end_time = endT
                adapOpt.timestepper_type = op.timestepper
                adapOpt.timestep = dt
                adapOpt.output_directory = di
                adapOpt.log_output = op.printStats
                adapOpt.export_diagnostics = True
                adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
                adapOpt.use_wetting_and_drying = op.wd
                if op.wd:
                    adapOpt.wetting_and_drying_alpha = alpha
                if mode == 'rossby-wave':
                    adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
                field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
                e = exporter.ExportManager(di + 'hdf5',
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
                    if mode == 'tohoku':
                        cb1 = err.TohokuCallback(adapSolver)
                    elif mode == 'shallow-water':
                        cb1 = err.ShallowWaterCallback(adapSolver)
                    elif mode == 'rossby-wave':
                        cb1 = err.RossbyWaveCallback(adapSolver)
                    if cnt != 0:
                        cb1.objective_functional = J_h
                    adapSolver.add_callback(cb1, 'timestep')
                solver_obj.bnd_functions['shallow_water'] = BCs
                adapSolver.iterate()
                if op.outputOF:
                    J_h = cb1.__call__()[1]  # Evaluate objective functional

                # Get mesh stats
                nEle = msh.meshStats(mesh_H)[0]
                mM = [min(nEle, mM[0]), max(nEle, mM[1])]
                Sn += nEle
                cnt += op.rm
                av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt*dt, dt)

            adaptTimer = clock() - adaptTimer
            msc.dis('Adaptive primal run complete. Run time: %.3fs' % adaptTimer, op.printStats)
    else:
        av = nEle
    if op.printStats:
        msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)

    # Measure error using metrics, using data from Huang et al.
    if mode == 'rossby-wave':
        index = int(cntT/op.ndump) if approach == 'fixedMesh' else int((cnt-op.rm) / op.ndump)
        with DumbCheckpoint(di+'hdf5/Elevation2d_'+msc.indexString(index), mode=FILE_READ) as loadElev:
            loadElev.load(elev_2d, name='elev_2d')
            loadElev.close()
        # peak_i, peak = msc.getMax(inte.interp(adap.isoP2(mesh_H), elev_2d)[0].dat.data)
        peak_i, peak = msc.getMax(elev_2d.dat.data)
        dgCoords = Function(VectorFunctionSpace(mesh_H, op.space2, op.degree2)).interpolate(mesh_H.coordinates)
        distanceTravelled = np.abs(dgCoords.dat.data[peak_i][0])

    toc = clock() - tic
    if mode == 'rossby-wave':
        return av, np.abs(peak/0.1567020), distanceTravelled, distanceTravelled/47.18, toc
    else:
        return av, np.abs(op.J(mode) - J_h)/np.abs(op.J(mode)), J_h, toc

    # TODO: Also generate and output a timeseries plot for the integrand of the objective functional [Anca Belme paper]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}")
    parser.add_argument("approach", help="Choose error estimator from {'norm', 'fieldBased', 'gradientBased', "
                                         "'hessianBased', 'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', "
                                         "'DWR', 'DWE'}" )
    parser.add_argument("-w", help="Use wetting and drying")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    print("Approach: ", args.approach)
    mode = args.mode

    # Choose mode and set parameter values
    approach, getData, getError, useAdjoint, aposteriori = msc.cheatCodes(args.approach)
    op = opt.Options(mode=mode,
                     vscale=0.1 if approach in ('DWR', 'gradientBased') else 0.85,
                     family='dg-dg',
                     rm=60 if useAdjoint else 30,
                     gradate=True if aposteriori else False,
                     advect=False,
                     window=True if approach == 'DWF' else False,
                     outputMetric=False,
                     plotpvd=True,
                     gauges=False,
                     bootstrap=False,
                     printStats=True,
                     outputOF=True,
                     orderChange=1 if approach in ('explicit', 'DWR', 'residual') else 0,
                     # orderChange=0,
                     wd=True if args.w else False,
                     ndump=10)
    if mode == 'shallow-water':
        op.rm = 20 if useAdjoint else 10
    elif mode == 'rossby-wave':
        op.rm = 48 if useAdjoint else 24

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
        # for i in range(1, 6):
        for i in range(4, 5):
            if mode == 'rossby-wave':
                av, relativePeak, distanceTravelled, phaseSpd, timing = \
                    solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
                print('Run %d: <#Elements>: %6d  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm  Timing %.1fs'
                      % (i, av, relativePeak, distanceTravelled, phaseSpd, timing))
                textfile.write('%d, %.4f, %.4f, %.4f, %.1f\n' % (av, relativePeak, distanceTravelled, phaseSpd, timing))
            else:
                av, rel, J_h, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
                print('Run %d: Mean element count %6d Relative error %.4e Timing %.1fs'
                      % (i, av, rel, timing))
                textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, timing, J_h))
    textfile.close()
