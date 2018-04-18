from thetis_adjoint import *
from thetis.field_defs import field_metadata
from firedrake_adjoint import *
import pyadjoint
from fenics_adjoint.solving import SolveBlock

import numpy as np
from time import clock
import datetime

from utils.adaptivity import isoP2, constructGradient, isotropicMetric, steadyMetric, metricIntersection, metricGradation
from utils.callbacks import *
from utils.error import explicitErrorEstimator, fluxJumpError
from utils.forms import formsSW, interelementTerm, strongResidualSW, solutionRW
from utils.interpolation import *
from utils.mesh import problemDomain, meshStats
from utils.misc import *
from utils.options import *

now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)


def solverSW(startRes, approach, getData, getError, useAdjoint, aposteriori, mode='tohoku', op=Options()):
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
    fullTime = clock()
    try:
        assert(mode in ('tohoku', 'shallow-water', 'rossby-wave'))
    except:
        raise NotImplementedError
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    primalTimer = dualTimer = errorTimer = adaptTimer = False

    # Establish filenames
    di = 'plots/'+mode+'/'
    if op.plotpvd:
        residualFile = File(di + "residual.pvd")
        implicitErrorFile = File(di + "implicitError.pvd")
        errorFile = File(di + "errorIndicator.pvd")
        adjointFile = File(di + "adjoint.pvd")

    # Setup problem
    mesh_H, u0, eta0, b, BCs, f = problemDomain(mode, startRes, op=op)
    V = op.mixedSpace(mesh_H)
    q = Function(V)
    uv_2d, elev_2d = q.split()  # Needed to load data into
    uv_2d.rename("uv_2d")
    elev_2d.rename("elev_2d")
    P1 = FunctionSpace(mesh_H, "CG", 1)
    if approach in ('residual', 'implicit', 'DWR', 'DWE', 'explicit'):
        q_ = Function(V)        # Variable at previous timestep
        uv_2d_, elev_2d_ = q_.split()
    P0 = FunctionSpace(mesh_H, "DG", 0)
    if mode == 'rossby-wave':
        peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])   # Analytic final-time state

    # Define Functions relating to a posteriori estimators
    if aposteriori or approach == 'norm':
        if useAdjoint:                          # Define adjoint variables
            dual = Function(V)
            dual_u, dual_e = dual.split()
            dual_u.rename("Adjoint velocity")
            dual_e.rename("Adjoint elevation")
        if op.orderChange:                      # Define variables on higher/lower order space
            Ve = op.mixedSpace(mesh_H, orderChange=op.orderChange)
            qe = Function(Ve)
            ue, ee = qe.split()
            qe_ = Function(Ve)
            ue_, ee_ = qe_.split()
            if useAdjoint:
                duale = Function(Ve)
                duale_u, duale_e = duale.split()
            be = Function(FunctionSpace(mesh_H, "CG", 1+op.orderChange)).assign(float(np.average(b.dat.data)))
        elif op.refinedSpace:                   # Define variables on refined space
            mesh_h = isoP2(mesh_H)
            Ve = op.mixedSpace(mesh_h)
            if mode == 'tohoku':
                be = TohokuDomain(mesh=mesh_h)[2]
            else:
                be = Function(FunctionSpace(mesh_h, "CG", 1)).assign(float(np.average(b.dat.data)))
            qe = Function(Ve)
            P0 = FunctionSpace(mesh_h, "DG", 0)
        else:                                   # Copy standard variables to mimic enriched space labels
            Ve = V
            qe = q
            be = b
            if approach != 'DWP':
                qe_ = q_
        if approach in ('implicit', 'DWE'):     # Define variables for implicit error estimation
            e_ = Function(Ve)
            e = Function(Ve)
            e0, e1 = e.split()
            e0.rename("Implicit error 0")
            e1.rename("Implicit error 1")
            et = TestFunction(Ve)
            (et0, et1) = (as_vector((et[0], et[1])), et[2])
            normal = FacetNormal(mesh_H)
        elif approach in ('residual', 'explicit', 'DWR'):
            rho = Function(Ve)
            rho_u, rho_e = rho.split()
            rho_u.rename("Velocity residual")
            rho_e.rename("Elevation residual")
    v = TestFunction(P0)

    # Initialise parameters and counters
    nEle, op.nVerT = meshStats(mesh_H)
    op.nVerT *= op.rescaling                # Target #Vertices
    mM = [nEle, nEle]                       # Min/max #Elements
    Sn = nEle
    endT = 0.
    dt = op.dt
    Dt = Constant(dt)
    T = op.Tend
    iStart = int(op.Tstart / dt)
    iEnd = int(T / (dt * op.rm)) if aposteriori and approach != 'DWP' else int(T / (dt * op.ndump))

    # Get initial boundary metric and TODO wetting and drying parameter
    if op.gradate or op.wd:
        H0 = Function(P1).interpolate(CellSize(mesh_H))
    if op.wd:
        g = constructGradient(elev_2d)
        spd = assemble(v * sqrt(inner(g, g)) * dx)
        gs = np.min(np.abs(spd.dat.data))
        print('#### gradient = ', gs)
        ls = np.min([H0.dat.data[i] for i in DirichletBC(P1, 0, 'on_boundary').nodes])
        print('#### ls = ', ls)
        alpha = Constant(gs * ls)
        print('#### alpha = ', alpha.dat.data)
        # alpha = Constant(0.5)
        # exit(23)

    tic = clock()
    if getData:
        if op.printStats:
            print('Starting fixed mesh primal run (forwards in time)')

        # Get solver parameter values and construct solver
        solver_obj = solver2d.FlowSolver2d(mesh_H, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True if op.nonlinear else False
        options.use_grad_depth_viscosity_term = False
        options.use_grad_div_viscosity_term = False
        if mode == 'rossby-wave':
            options.coriolis_frequency = f
        options.simulation_export_time = dt * (op.rm-1) if aposteriori and approach != 'DWP' else dt * op.ndump
        options.simulation_end_time = T
        options.period_of_interest_start = op.Tstart
        options.period_of_interest_end = T
        options.timestepper_type = op.timestepper
        options.timestep = dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.log_output = op.printStats
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        options.use_wetting_and_drying = op.wd
        if op.wd:
            options.wetting_and_drying_alpha = alpha
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        if mode == 'rossby-wave':
            cb1 = RossbyWaveCallback(solver_obj)
            cb2 = ObjectiveRWCallback(solver_obj)
        elif mode == 'shallow-water':
            cb1 = ShallowWaterCallback(solver_obj)
            cb2 = ObjectiveSWCallback(solver_obj)
        else:
            cb1 = TohokuCallback(solver_obj)
            cb2 = ObjectiveTohokuCallback(solver_obj)
            if approach == 'fixedMesh':
                cb3 = P02Callback(solver_obj)
                cb4 = P06Callback(solver_obj)
                solver_obj.add_callback(cb3, 'timestep')
                solver_obj.add_callback(cb4, 'timestep')
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.add_callback(cb2, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        if aposteriori and approach != 'DWP':
            def selector():
                rm = options.timesteps_per_remesh
                dt = options.timestep
                options.simulation_export_time = dt if int(solver_obj.simulation_time / dt) % rm == 0 else (rm - 1) * dt
        else:
            selector = None
        primalTimer = clock()
        solver_obj.iterate(export_func=selector)
        primalTimer = clock() - primalTimer
        J_h = cb1.quadrature()      # Evaluate objective functional
        integrand = cb1.__call__()[1]
        if op.printStats:
            print('Primal run complete. Run time: %.3fs' % primalTimer)
        if mode == 'tohoku' and approach == 'fixedMesh':
            totalVarP02 = cb3.totalVariation()
            totalVarP06 = cb4.totalVariation()

        # Reset counters
        cntT = int(np.ceil(T/dt))
        if useAdjoint:
            if op.printStats:
                print('\nGenerating dual solutions...')

            # Assemble objective functional
            Jfuncs = cb2.__call__()[1]
            J = 0
            for i in range(1, len(Jfuncs)):
                J += 0.5 * (Jfuncs[i - 1] + Jfuncs[i]) * dt

            # Compute gradient
            dualTimer = clock()
            dJdb = compute_gradient(J, Control(b))      # TODO: Rewrite pyadjoint code to avoid computing this
            # File(di + 'gradient.pvd').write(dJdb)     # Too memory intensive in Tohoku case
            print("Norm of gradient = %e" % dJdb.dat.norm)

            # Extract adjoint solutions
            tape = get_working_tape()
            solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
            N = len(solve_blocks)
            diff = op.ndump if approach == 'DWP' else op.rm
            r = N % diff   # Number of extra tape annotations in setup
            for i in range(N-1, r-2, -diff):
                dual.assign(solve_blocks[i].adj_sol)
                dual_u, dual_e = dual.split()
                dual_u.rename('Adjoint velocity')
                dual_e.rename('Adjoint elevation')
                with DumbCheckpoint(di+'hdf5/adjoint_'+indexString(int((i-r+1)/diff)), mode=FILE_CREATE) as saveAdj:
                    saveAdj.store(dual_u)
                    saveAdj.store(dual_e)
                    saveAdj.close()
                if op.plotpvd:
                    adjointFile.write(dual_u, dual_e, time=dt * (i-r+1))
                if op.printStats:
                    print('Adjoint simulation %.2f%% complete' % ((N - i + r - 1) / N * 100))
            dualTimer = clock() - dualTimer
            if op.printStats:
                print('Dual run complete. Run time: %.3fs' % dualTimer)
    cnt = 0

    # Loop back over times to generate error estimators
    if getError:
        if op.printStats:
            print('\nStarting error estimate generation')
        errorTimer = clock()

        # Define implicit error problem
        if approach in ('implicit', 'DWE'): # TODO: Check out situation with nonlinear option
            B_, L = formsSW(qe, qe_, b, Dt, impermeable=False, coriolisFreq=f, op=op)
            B = formsSW(e, e_, b, Dt, impermeable=False, coriolisFreq=f, op=op)[0]
            I = interelementTerm(et1 * ue, n=normal) * dS
            errorProblem = NonlinearVariationalProblem(B - L + B_ - I, e)
            errorSolver = NonlinearVariationalSolver(errorProblem, solver_parameters=op.params)

        for k in range(0, iEnd):
            if op.printStats:
                print('Generating error estimate %d / %d' % (k+1, iEnd))

            if approach == 'DWP':
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(k), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(k), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
            else:
                i1 = 0 if k == 0 else 2*k-1
                i2 = 2*k
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(i1), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(i1), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
                uv_2d_.assign(uv_2d)
                elev_2d_.assign(elev_2d)
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(i2), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(i2), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()

            # Solve implicit error problem
            if approach in ('implicit', 'DWE') or op.orderChange:
                ue.interpolate(uv_2d)
                ee.interpolate(elev_2d)
                ue_.interpolate(uv_2d_)
                ee_.interpolate(elev_2d_)
                if approach in ('implicit', 'DWE'):
                    errorSolver.solve()
                    e_.assign(e)
                    if approach == 'implicit':
                        epsilon = assemble(v * sqrt(inner(e, e)) * dx)
                    if op.plotpvd:
                        implicitErrorFile.write(e0, e1, time=float(k))

            # Approximate residuals
            if approach in ('explicit', 'residual', 'DWR'):
                if op.refinedSpace:
                    qe, qe_ = mixedPairInterp(mesh_h, Ve, q, q_)
                Au, Ae = strongResidualSW(qe, qe_, be, Dt, coriolisFreq=None, op=op)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=float(k))
                if approach == 'residual':
                    epsilon = assemble(v * sqrt(inner(rho, rho)) * dx)
                elif approach == 'explicit':
                    epsilon = explicitErrorEstimator(qe, rho, be if op.orderChange else be, v, maxBathy=True)

            if useAdjoint:
                with DumbCheckpoint(di+'hdf5/adjoint_'+indexString(k), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_u)
                    loadAdj.load(dual_e)
                    loadAdj.close()
                if approach == 'DWR':
                    if op.orderChange:
                        duale_u.interpolate(dual_u)
                        duale_e.interpolate(dual_e)
                        epsilon = assemble(v * inner(rho, duale) * dx)
                    elif op.refinedSpace:
                        duale = mixedPairInterp(mesh_h, dual)
                        epsilon = assemble(v * inner(rho, duale) * dx)
                    else:
                        epsilon = assemble(v * inner(rho, dual) * dx)
                elif approach == 'DWE':
                    if op.orderChange:
                        duale_u.interpolate(dual_u)
                        duale_e.interpolate(dual_e)
                        epsilon = assemble(v * inner(e, duale) * dx)
                    elif op.refinedSpace:
                        duale = mixedPairInterp(mesh_h, dual)
                        epsilon = assemble(v * inner(e, duale) * dx)
                    else:
                        epsilon = assemble(v * inner(e, dual) * dx)
                elif approach == 'DWP':
                    epsilon = assemble(v * inner(q, dual) * dx)
                    for i in range(k, min(k + iEnd - iStart, iEnd)):
                        with DumbCheckpoint(di + 'hdf5/adjoint_' + indexString(i), mode=FILE_READ) as loadAdj:
                            loadAdj.load(dual_u)
                            loadAdj.load(dual_e)
                            loadAdj.close()
                        epsilon_ = assemble(v * inner(q, dual) * dx)
                        for j in range(len(epsilon.dat.data)):
                            epsilon.dat.data[j] = max(epsilon.dat.data[j], epsilon_.dat.data[j])

            # Store error estimates
            epsilon.rename("Error indicator")
            with DumbCheckpoint(di+'hdf5/'+approach+'Error'+indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        if op.printStats:
            print('Errors estimated. Run time: %.3fs' % errorTimer)

    if approach != 'fixedMesh':
        with pyadjoint.stop_annotating():
            q = Function(V)
            uv_2d, elev_2d = q.split()
            elev_2d.interpolate(eta0)
            uv_2d.interpolate(u0)
            if aposteriori:
                epsilon = Function(P0, name="Error indicator")
            if op.printStats:
                print('\nStarting adaptive mesh primal run (forwards in time)')
            adaptTimer = clock()
            while cnt < int(T / dt):
                stepTimer = clock()
                indexStr = indexString(int(cnt/op.ndump))

                # Load variables from disk
                if cnt != 0:
                    V = op.mixedSpace(mesh_H)
                    q = Function(V)
                    uv_2d, elev_2d = q.split()
                    with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexStr, mode=FILE_READ) as loadElev:
                        loadElev.load(elev_2d, name='elev_2d')
                        loadElev.close()
                    with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexStr, mode=FILE_READ) as loadVel:
                        loadVel.load(uv_2d, name='uv_2d')
                        loadVel.close()

                for l in range(op.nAdapt):      # TODO: Test this functionality

                    # Construct metric
                    if aposteriori:
                        with DumbCheckpoint(di+'hdf5/'+approach+'Error'+indexString(int(cnt/op.rm)), mode=FILE_READ) \
                                as loadErr:
                            loadErr.load(epsilon)
                            loadErr.close()
                        errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(interp(mesh_H, epsilon))
                        M = isotropicMetric(errEst, invert=False, op=op)    # TODO: Not sure normalisation is working
                    else:
                        if approach == 'norm':
                            v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                            epsilon = assemble(v * inner(q, q) * dx)
                            M = isotropicMetric(epsilon, invert=False, op=op)
                        elif approach =='fluxJump' and cnt != 0:
                            v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                            epsilon = fluxJumpError(q, v)
                            M = isotropicMetric(epsilon, invert=False, op=op)
                        else:
                            if op.adaptField != 's':
                                if approach == 'fieldBased':
                                    M = isotropicMetric(elev_2d, invert=False, op=op)
                                elif approach == 'gradientBased':
                                    M = isotropicMetric(constructGradient(elev_2d), invert=False, op=op)
                                elif approach == 'hessianBased':
                                    M = steadyMetric(elev_2d, op=op)
                            if cnt != 0:    # Can't adapt to zero velocity
                                if op.adaptField != 'f':
                                    spd = Function(FunctionSpace(mesh_H, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                                    if approach == 'fieldBased':
                                        M2 = isotropicMetric(spd, invert=False, op=op)
                                    elif approach == 'gradientBased':
                                        M2 = isotropicMetric(constructGradient(spd), invert=False, op=op)
                                    elif approach == 'hessianBased':
                                        M2 = steadyMetric(spd, op=op)
                                    M = metricIntersection(M, M2) if op.adaptField == 'b' else M2
                    if op.gradate:
                        M_ = isotropicMetric(interp(mesh_H, H0), bdy=True, op=op)  # Initial boundary metric
                        M = metricIntersection(M, M_, bdy=True)
                        metricGradation(M, op=op)
                    if op.plotpvd:
                        File('plots/'+mode+'/mesh.pvd').write(mesh_H.coordinates, time=float(cnt))

                    # Adapt mesh and interpolate variables
                    if not (((approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.adaptField != 'f')
                             or approach == 'fluxJump') and cnt == 0):
                        mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                        P1 = FunctionSpace(mesh_H, "CG", 1)
                        elev_2d, uv_2d = interp(mesh_H, elev_2d, uv_2d)
                        if mode == 'tohoku':
                            b = interp(mesh_H, b)
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
                adapOpt.period_of_interest_start = op.Tstart
                adapOpt.period_of_interest_end = T
                adapOpt.timestepper_type = op.timestepper
                adapOpt.timestep = dt
                adapOpt.output_directory = di
                adapOpt.export_diagnostics = True
                adapOpt.log_output = op.printStats
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
                if mode == 'rossby-wave':
                    cb1 = RossbyWaveCallback(adapSolver)
                elif mode == 'shallow-water':
                    cb1 = ShallowWaterCallback(adapSolver)
                else:
                    cb1 = TohokuCallback(adapSolver)
                    cb3 = P02Callback(adapSolver)
                    cb4 = P06Callback(adapSolver)
                if cnt != 0:
                    cb1.objective_value = integrand
                    if mode == 'tohoku':
                        cb3.gauge_values = gP02
                        cb4.gauge_values = gP06
                adapSolver.add_callback(cb1, 'timestep')
                if mode == 'tohoku':
                    adapSolver.add_callback(cb3, 'timestep')
                    adapSolver.add_callback(cb4, 'timestep')
                adapSolver.bnd_functions['shallow_water'] = BCs
                adapSolver.iterate()
                J_h = cb1.quadrature()
                integrand = cb1.__call__()[1]
                if mode == 'tohoku':
                    gP02 = cb3.__call__()[1]
                    gP06 = cb4.__call__()[1]

                # Get mesh stats
                nEle = meshStats(mesh_H)[0]
                mM = [min(nEle, mM[0]), max(nEle, mM[1])]
                Sn += nEle
                cnt += op.rm
                av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt*dt, dt)

            adaptTimer = clock() - adaptTimer
            if mode == 'tohoku':
                totalVarP02 = cb3.totalVariation()
                totalVarP06 = cb4.totalVariation()
            if op.printStats:
                print('Adaptive primal run complete. Run time: %.3fs' % adaptTimer)
    else:
        av = nEle
    fullTime = clock() - fullTime
    if op.printStats:
        printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, fullTime)

    if mode == 'rossby-wave':   # Measure error using metrics, using data from Huang et al.
        index = int(cntT/op.ndump) if approach == 'fixedMesh' else int((cnt-op.rm) / op.ndump)
        with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(index), mode=FILE_READ) as loadElev:
            loadElev.load(elev_2d, name='elev_2d')
            loadElev.close()
        peak, distance = peakAndDistance(elev_2d, op=op)
        print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

    toc = clock() - tic
    rel = np.abs(op.J(mode) - J_h) / np.abs(op.J(mode))
    if mode == 'rossby-wave':
        return av, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, toc
    elif mode == 'tohoku':
        return av, rel, J_h, integrand, totalVarP02, totalVarP06, toc
    else:
        return av, rel, J_h, integrand, toc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}")
    parser.add_argument("approach", help="Choose error estimator from {'norm', 'fieldBased', 'gradientBased', "
                                         "'hessianBased', 'residual', 'explicit', 'fluxJump', 'implicit', 'DWP', "
                                         "'DWR', 'DWE'}" )
    parser.add_argument("-w", help="Use wetting and drying")
    parser.add_argument("-ho", help="Compute errors and residuals in a higher order space")
    parser.add_argument("-lo", help="Compute errors and residuals in a lower order space")
    parser.add_argument("-r", help="Compute errors and residuals in a refined space")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-f", help="Field for adaption")
    args = parser.parse_args()
    if args.ho:
        assert (not args.r) and (not args.lo)
    if args.lo:
        assert (not args.r) and (not args.ho)
    if args.r:
        assert (not args.ho) and (not args.lo)
    if args.f:
        assert(args.f in ('s', 'f', 'b'))
        f = args.f
    else:
        f = 's'
    print("Mode: ", args.mode)
    print("Approach: ", args.approach)
    mode = args.mode

    # Choose mode and set parameter values
    approach, getData, getError, useAdjoint, aposteriori = cheatCodes(args.approach)
    op = Options(mode=mode,
                 rm=100 if useAdjoint else 50,
                 # gradate=True if aposteriori and mode == 'tohoku' else False,
                 gradate=False,     # TODO: Fix this for tohoku case
                 plotpvd=True if args.o else False,
                 printStats=True,
                 wd=True if args.w else False,
                 adaptField=f)
    if mode == 'shallow-water':
        op.rm = 10 if useAdjoint else 5
    elif mode == 'rossby-wave':
        op.rm = 48 if useAdjoint else 24
    op.nonlinear = False        # TODO: This is only temporary, while pyadjoint and nonlinear are incompatible

    # Establish filename
    filename = 'outdata/' + mode + '/' + approach
    if args.ho:
        op.orderChange = 1
        filename += '_ho'
    elif args.lo:
        op.orderChange = -1
        filename += '_lo'
    elif args.r:
        op.refinedSpace = True
        filename += '_r'
    if args.w:
        filename += '_w'
    filename += '_' + date
    textfile = open(filename + '.txt', 'w+')
    integrandFile = open(filename + 'Integrand.txt', 'w+')

    # Run simulations
    resolutions = range(1, 2)
    Jlist = np.zeros(len(resolutions))
    if mode == 'tohoku':
        g2list = np.zeros(len(resolutions))
        g6list = np.zeros(len(resolutions))
    for i in resolutions:       # TODO: Can't currently do multiple adjoint runs
        # Get data and save to disk
        if mode == 'rossby-wave':
            av, rel, J_h, integrand, relativePeak, distance, phaseSpd, tim = \
                solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
            print('Run %d: <#Elements>: %6d Obj. error: %.4e  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm  Timing %.1fs'
                  % (i, av, rel, relativePeak, distance, phaseSpd, tim))
            textfile.write('%d, %.4e, %.4f, %.4f, %.4f, %.1f, %.4e\n'
                           % (av, rel, relativePeak, distance, phaseSpd, tim, J_h))
        elif mode == 'tohoku':
            av, rel, J_h, integrand, totalVarP02, totalVarP06, tim = solverSW(i, approach, getData, getError,
                                                                              useAdjoint, aposteriori, mode=mode, op=op)
            print('Run %d: Mean element count %6d Relative error %.4e P02: %.3f P06: %.3f Timing %.1fs'
                  % (i, av, rel, totalVarP02, totalVarP06, tim))
            textfile.write('%d, %.4e, %.3f, %.3f, %.1f, %.4e\n' % (av, rel, totalVarP02, totalVarP06, tim, J_h))
        else:
            av, rel, J_h, integrand, tim = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode,
                                                    op=op)
            print('Run %d: Mean element count %6d Relative error %.4e Timing %.1fs'
                  % (i, av, rel, tim))
            textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, tim, J_h))
        integrandFile.writelines(["%s," % val for val in integrand])
        integrandFile.write("\n")

        # Calculate orders of convergence
        if not useAdjoint:                              # TODO: Get around this
            Jlist[i] = J_h
            convList = np.zeros(len(resolutions) - 2)   # TODO
            if mode == 'tohoku':
                g2list[i] = totalVarP02
                g6list[i] = totalVarP06
            if i > 1:
                Jconv = (Jlist[i] - Jlist[i - 1]) / (Jlist[i - 1] - Jlist[i - 2])
                if mode == 'tohoku':
                    g2conv = (g2list[i] - g2list[i - 1]) / (g2list[i - 1] - g2list[i - 2])
                    g6conv = (g6list[i] - g6list[i - 1]) / (g6list[i - 1] - g6list[i - 2])
                    print("Orders of convergence... J: %.4f, P02: %.4f, P06: %.4f" % (Jconv, g2conv, g6conv))
                else:
                    print("Order of convergence: %.4f" % Jconv)
    textfile.close()
    integrandFile.close()
