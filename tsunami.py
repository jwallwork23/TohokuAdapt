from thetis_adjoint import *
from thetis.field_defs import field_metadata
from thetis.shallowwater_eq import ShallowWaterMomentumEquation, FreeSurfaceEquation    # For residual computation
import pyadjoint
from fenics_adjoint.solving import SolveBlock                                       # For extracting adjoint solutions

import numpy as np
from time import clock

from utils.adaptivity import isoP2, isotropicMetric, metricIntersection, metricGradation, pointwiseMax, steadyMetric
from utils.callbacks import *
from utils.forms import strongResidualSW
from utils.interpolation import interp, mixedPairInterp
from utils.setup import problemDomain, solutionRW
from utils.misc import indexString, peakAndDistance, meshStats
from utils.options import Options


def fixedMesh(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/fixedMesh/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        nEle = meshStats(mesh)[0]
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        if op.mode == 'rossby-wave':
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state

        # Initialise solver
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.use_grad_depth_viscosity_term = False           # TODO: Might as well include this for Tohoku?
        options.use_lax_friedrichs_velocity = False             # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.ndump
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = SWCallback(solver_obj)
        cb1.op = op
        if op.mode == 'tohoku':
            cb2 = P02Callback(solver_obj)
            cb3 = P06Callback(solver_obj)
            solver_obj.add_callback(cb2, 'timestep')
            solver_obj.add_callback(cb3, 'timestep')
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs

        # Solve and extract timeseries / functionals
        solverTimer = clock()
        solver_obj.iterate()
        solverTimer = clock() - solverTimer
        J_h = cb1.quadrature()          # Evaluate objective functional
        integrand = cb1.__call__()[1]   # and get integrand timeseries
        if op.mode == 'tohoku':
            totalVarP02 = cb2.totalVariation()
            totalVarP06 = cb3.totalVariation()

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT/op.ndump)
            with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return nEle, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, solverTimer
        elif op.mode == 'tohoku':
            return nEle, rel, J_h, integrand, totalVarP02, totalVarP06, solverTimer
        else:
            return nEle, rel, J_h, integrand, solverTimer


def hessianBased(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/hessianBased/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        V = op.mixedSpace(mesh)
        P1 = FunctionSpace(mesh, "CG", 1)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        if op.mode == 'rossby-wave':
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)

        # Initialise parameters and counters
        nEle, op.nVerT = meshStats(mesh)
        op.nVerT *= op.rescaling  # Target #Vertices
        mM = [nEle, nEle]  # Min/max #Elements
        Sn = nEle
        cnt = 0
        endT = 0.

        adaptSolveTimer = 0.
        while cnt < op.cntT:
            indexStr = indexString(int(cnt / op.ndump))

            # Load variables from disk
            if cnt != 0:
                V = op.mixedSpace(mesh)
                q = Function(V)
                uv_2d, elev_2d = q.split()
                with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            adaptTimer = clock()
            for l in range(op.nAdapt):                                          # TODO: Test this functionality

                # Construct metric
                if op.adaptField != 's':
                    M = steadyMetric(elev_2d, op=op)
                if cnt != 0:  # Can't adapt to zero velocity
                    if op.adaptField != 'f':
                        spd = Function(FunctionSpace(mesh, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                        M2 = steadyMetric(spd, op=op)
                        M = metricIntersection(M, M2) if op.adaptField == 'b' else M2
                if op.bAdapt:
                    M2 = steadyMetric(b, op=op)
                    M = M2 if op.adaptField != 'f' and cnt == 0. else metricIntersection(M, M2)

                # Adapt mesh and interpolate variables
                if op.bAdapt or cnt != 0 or op.adaptField == 'f':
                    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                    P1 = FunctionSpace(mesh, "CG", 1)
                    elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
                    b = problemDomain(mesh=mesh, op=op)[3]                      # Reset bathymetry on new mesh
                    uv_2d.rename('uv_2d')
                    elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.use_grad_depth_viscosity_term = False               # TODO: Might as well include this for Tohoku?
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = di
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh)[1])
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

            # Establish callbacks and iterate
            cb1 = SWCallback(adapSolver)
            cb1.op = op
            if op.mode == 'tohoku':
                cb2 = P02Callback(adapSolver)
                cb3 = P06Callback(adapSolver)
                if cnt == 0:
                    initP02 = cb2.init_value
                    initP06 = cb3.init_value
            adapSolver.add_callback(cb1, 'timestep')
            if cnt != 0:
                cb1.objective_value = integrand
                if op.mode == 'tohoku':
                    cb2.gauge_values = gP02
                    cb2.init_value = initP02
                    cb3.gauge_values = gP06
                    cb3.init_value = initP06
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode == 'tohoku':
                adapSolver.add_callback(cb2, 'timestep')
                adapSolver.add_callback(cb3, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            J_h = cb1.quadrature()
            integrand = cb1.__call__()[1]
            if op.mode == 'tohoku':
                gP02 = cb2.__call__()[1]
                gP06 = cb3.__call__()[1]
                totalVarP02 = cb2.totalVariation()
                totalVarP06 = cb3.totalVariation()

            # Get mesh stats
            nEle = meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt/op.rm+1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)

            adaptSolveTimer += adaptTimer + solverTimer

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT / op.ndump)
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return av, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, adaptSolveTimer
        elif op.mode == 'tohoku':
            return av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, adaptSolveTimer
        else:
            return av, rel, J_h, integrand, adaptSolveTimer


def DWR(startRes, op=Options()):
    initTimer = clock()
    di = 'plots/' + op.mode + '/DWR/'
    if op.plotpvd:
        residualFile = File(di + "residual.pvd")
        errorFile = File(di + "errorIndicator.pvd")
        adjointFile = File(di + "adjoint.pvd")

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh_H, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh_H)
    q = Function(V)
    uv_2d, elev_2d = q.split()    # Needed to load data into
    uv_2d.rename('uv_2d')
    elev_2d.rename('elev_2d')
    P1 = FunctionSpace(mesh_H, "CG", 1)
    q_ = Function(V)                        # Variable at previous timestep
    uv_2d_, elev_2d_ = q_.split()
    P0 = FunctionSpace(mesh_H, "DG", 0)
    if op.mode == 'rossby-wave':
        peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
    if op.orderChange:                      # Define variables on higher/lower order space
        Ve = op.mixedSpace(mesh_H)          # Automatically generates a higher/lower order space
        duale = Function(Ve)
        duale_u, duale_e = duale.split()
        be = b
    elif op.refinedSpace:                   # Define variables on an iso-P2 refined space
        mesh_h = isoP2(mesh_H)
        Ve = op.mixedSpace(mesh_h)
        be = problemDomain(mesh=mesh_h, op=op)[3]
        P0 = FunctionSpace(mesh_h, "DG", 0)
    else:                                   # Copy standard variables to mimic enriched space labels
        Ve = V
        be = b
    qe = Function(Ve)
    qe_ = Function(Ve)
    rho = Function(Ve)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    v = TestFunction(P0)                    # For extracting elementwise error indicators

    # Initialise parameters and counters
    nEle, op.nVerT = meshStats(mesh_H)
    op.nVerT *= op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    endT = 0.

    # Get initial boundary metric
    if op.gradate:
        H0 = Function(P1).interpolate(CellSize(mesh_H))

    if not op.regen:    # TODO: regen won't work in general as HDF5 files get overwritten in adaptive stage

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh_H, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.use_grad_depth_viscosity_term = False                   # TODO: Might as well include this for Tohoku?
        options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.dt * (op.rm - 1)
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        def selector():
            rm = options.timesteps_per_remesh
            dt = options.timestep
            options.simulation_export_time = dt if int(solver_obj.simulation_time / dt) % rm == 0 else (rm - 1) * dt
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)
        primalTimer = clock()
        solver_obj.iterate(export_func=selector)
        primalTimer = clock() - primalTimer
        J = cb1.assembleOF()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        dJdb = compute_gradient(J, Control(b))      # TODO: Rewrite pyadjoint to avoid computing this
        gradientTimer = clock() - gradientTimer
        # File(di + 'gradient.pvd').write(dJdb)     # Too memory intensive in Tohoku case
        print("Norm of gradient: %.3e. Computation time: %.1fs" % (dJdb.dat.norm, gradientTimer))

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        dualTimer = clock() - dualTimer
        r = N % op.rm  # Number of extra tape annotations in setup
        print('Dual run complete. Run time: %.3fs' % dualTimer)

    with pyadjoint.stop_annotating():

        # # TODO: For updated residual computation
        # v, xi = TestFunctions(Ve)
        # swe_mom = ShallowWaterMomentumEquation(v, Ve.sub(0), Ve.sub(1), be, solver_obj.options)
        # swe_con = FreeSurfaceEquation(xi, Ve.sub(1), Ve.sub(0), be, solver_obj.options)

        errorTimer = clock()
        for i, k in zip(range(r-1, N-1, op.rm), range(0, int(op.cntT / op.rm))):
            print('Generating error estimate %d / %d' % (k + 1, int(op.cntT / op.rm)))
            i1 = 0 if k == 0 else 2 * k - 1
            i2 = 2 * k
            with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexString(i1), mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d)
                loadVel.close()
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(i1), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d)
                loadElev.close()
            uv_2d_.assign(uv_2d)
            elev_2d_.assign(elev_2d)
            with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexString(i2), mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d)
                loadVel.close()
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(i2), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d)
                loadElev.close()
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            if op.plotpvd:
                adjointFile.write(dual_u, dual_e, time=float(op.dt * op.rm * k))

            # Approximate residuals, load adjoint data and form residuals
            if op.refinedSpace:
                qe, qe_ = mixedPairInterp(mesh_h, Ve, q, q_)

            # Initial approach  # TODO: Replace with actual Thetis forms: see other TODOs
            Au, Ae = strongResidualSW(qe, qe_, be, coriolisFreq=None, op=op)
            rho_u.interpolate(Au)
            rho_e.interpolate(Ae)

            # # Attempt at updated approach. Although residuals here are weak
            # uv_2d_e, elev_2d_e = qe.split()     # TODO: Replace older version, when complete. Currently not local
            # uv_2d_e_, elev_2d_e_ = qe_.split()
            # rho_u = swe_mom.residual('all',     # Terms used, from {'all', 'source', 'implicit', 'explicit', 'nonlinear'}
            #                          uv_2d_e,
            #                          uv_2d_e_,
            #                          {'eta': elev_2d_e, 'uv': uv_2d_e},
            #                          {'eta': elev_2d_e_, 'uv': uv_2d_e_},
            #                          BCs)
            # rho_e = swe_con.residual('all',
            #                          elev_2d_e,
            #                          elev_2d_e_,
            #                          {'eta': elev_2d_e, 'uv': uv_2d_e},
            #                          {'eta': elev_2d_e_, 'uv': uv_2d_e_},
            #                          BCs)

            if op.plotpvd:
                residualFile.write(rho_u, rho_e, time=float(op.dt * op.rm * k))
            if op.orderChange:
                duale_u.interpolate(dual_u)
                duale_e.interpolate(dual_e)
                epsilon = assemble(v * inner(rho, duale) * dx)
            elif op.refinedSpace:
                duale = mixedPairInterp(mesh_h, dual)
                epsilon = assemble(v * inner(rho, duale) * dx)
            else:
                # epsilon = assemble(v * inner(rho, dual) * dx)
                epsilon = assemble(v * (inner(rho_u, dual_u) + inner(rho_e, dual_e)) * dx)
            epsilon.rename("Error indicator")
            with DumbCheckpoint(di + 'hdf5/Error2d_' + indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        print('Errors estimated. Run time: %.3fs' % errorTimer)

        # Run adaptive primal run
        cnt = 0
        adaptSolveTimer = 0.
        q = Function(V)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        while cnt < op.cntT:
            indexStr = indexString(int(cnt / op.ndump))

            # Load variables from solver session on previous mesh
            adaptTimer = clock()
            if cnt != 0:
                V = op.mixedSpace(mesh_H)
                q = Function(V)
                uv_2d, elev_2d = q.split()
                with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            for l in range(op.nAdapt):                                          # TODO: Test this functionality

                # Construct metric
                with DumbCheckpoint(di + 'hdf5/Error2d_' + indexString(int(cnt / op.rm)), mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(interp(mesh_H, epsilon))
                M = isotropicMetric(errEst, invert=False, op=op)                # TODO: Not sure normalisation is working
                if op.gradate:
                    M_ = isotropicMetric(interp(mesh_H, H0), bdy=True, op=op)   # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=True)
                    metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                P1 = FunctionSpace(mesh_H, "CG", 1)
                elev_2d, uv_2d = interp(mesh_H, elev_2d, uv_2d)
                b = problemDomain(mesh=mesh_H, op=op)[3]                        # Reset bathymetry on new mesh
                uv_2d.rename('uv_2d')
                elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh_H, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.use_grad_depth_viscosity_term = False               # TODO: Might as well include this for Tohoku?
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = di
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
            e = exporter.ExportManager(di + 'hdf5',
                                       ['elev_2d', 'uv_2d'],
                                       {'elev_2d': elev_2d, 'uv_2d': uv_2d},
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
            cb2 = SWCallback(adapSolver)
            cb2.op = op
            if op.mode == 'tohoku':
                cb3 = P02Callback(adapSolver)
                cb4 = P06Callback(adapSolver)
                if cnt == 0:
                    initP02 = cb3.init_value
                    initP06 = cb4.init_value
            if cnt != 0:
                cb2.objective_value = integrand
                if op.mode == 'tohoku':
                    cb3.gauge_values = gP02
                    cb3.init_value = initP02
                    cb4.gauge_values = gP06
                    cb4.init_value = initP06
            adapSolver.add_callback(cb2, 'timestep')
            if op.mode == 'tohoku':
                adapSolver.add_callback(cb3, 'timestep')
                adapSolver.add_callback(cb4, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            J_h = cb2.quadrature()
            integrand = cb2.__call__()[1]
            if op.mode == 'tohoku':
                gP02 = cb3.__call__()[1]
                gP06 = cb4.__call__()[1]

            # Get mesh stats
            nEle = meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)

            adaptSolveTimer += adaptTimer + solverTimer

        if op.mode == 'tohoku':
            totalVarP02 = cb3.totalVariation()
            totalVarP06 = cb4.totalVariation()

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT / op.ndump)
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return av, rel, J_h, integrand, np.abs(
                peak / peak_a), distance, distance / distance_a, adaptSolveTimer
        elif op.mode == 'tohoku':
            return av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, adaptSolveTimer
        else:
            return av, rel, J_h, integrand, adaptSolveTimer


def DWP(startRes, op=Options()):
    initTimer = clock()
    di = 'plots/' + op.mode + '/DWP/'
    if op.plotpvd:
        errorFile = File(di + "errorIndicator.pvd")
        adjointFile = File(di + "adjoint.pvd")

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh_H, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh_H)
    q = Function(V)
    uv_2d, elev_2d = q.split()  # Needed to load data into
    uv_2d.rename('uv_2d')
    elev_2d.rename('elev_2d')
    if op.mode == 'rossby-wave':
        peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
    v = TestFunction(FunctionSpace(mesh_H, "DG", 0))  # For extracting elementwise error indicators

    # Initialise parameters and counters
    nEle, op.nVerT = meshStats(mesh_H)
    op.nVerT *= op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    endT = 0.

    # Get initial boundary metric
    if op.gradate:
        H0 = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(CellSize(mesh_H))

    if not op.regen:    # TODO: regen won't work in general as HDF5 files get overwritten in adaptive stage

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh_H, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.use_grad_depth_viscosity_term = False                   # TODO: Might as well include this for Tohoku?
        options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.ndump
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)
        primalTimer = clock()
        solver_obj.iterate()
        primalTimer = clock() - primalTimer
        J = cb1.assembleOF()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        dJdb = compute_gradient(J, Control(b))      # TODO: Rewrite pyadjoint to avoid computing this
        gradientTimer = clock() - gradientTimer
        # File(di + 'gradient.pvd').write(dJdb)     # Too memory intensive in Tohoku case
        print("Norm of gradient: %.3e. Computation time: %.1fs" % (dJdb.dat.norm, gradientTimer))

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.ndump                            # Number of extra tape annotations in setup
        for i in range(N - 1, r - 2, -op.ndump):
            dual.assign(solve_blocks[i].adj_sol)    # TODO: Could this be combined with error estimation step?
            dual_u, dual_e = dual.split()
            dual_u.rename('Adjoint velocity')
            dual_e.rename('Adjoint elevation')
            with DumbCheckpoint(di + 'hdf5/Adjoint2d_' + indexString(int((i - r + 1) / op.ndump)), mode=FILE_CREATE) as saveAdj:
                saveAdj.store(dual_u)
                saveAdj.store(dual_e)
                saveAdj.close()
            if op.plotpvd:
                adjointFile.write(dual_u, dual_e, time=op.dt * (i - r + 1))
            print('Adjoint simulation %.2f%% complete' % ((N - i + r - 1) / N * 100))
        dualTimer = clock() - dualTimer
        print('Dual run complete. Run time: %.3fs' % dualTimer)

    with pyadjoint.stop_annotating():

        errorTimer = clock()
        for k in range(0, op.iEnd):  # Loop back over times to generate error estimators
            print('Generating error estimate %d / %d' % (k + 1, op.iEnd))
            with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexString(k), mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d)
                loadVel.close()
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(k), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d)
                loadElev.close()

            # Load adjoint data and form indicators
            epsilon = assemble(v * inner(q, dual) * dx)
            for i in range(k, min(k + op.iEnd - op.iStart, op.iEnd)):
                with DumbCheckpoint(di + 'hdf5/Adjoint2d_' + indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_u)
                    loadAdj.load(dual_e)
                    loadAdj.close()
                epsilon = pointwiseMax(epsilon, assemble(v * inner(q, dual) * dx))
            epsilon.rename("Error indicator")
            with DumbCheckpoint(di + 'hdf5/Error2d_' + indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        print('Errors estimated. Run time: %.3fs' % errorTimer)

        # Run adaptive primal run
        cnt = 0
        adaptSolveTimer = 0.
        q = Function(V)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        while cnt < op.cntT:
            indexStr = indexString(int(cnt / op.ndump))

            # Load variables from solver session on previous mesh
            adaptTimer = clock()
            if cnt != 0:
                V = op.mixedSpace(mesh_H)
                q = Function(V)
                uv_2d, elev_2d = q.split()
                with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            for l in range(op.nAdapt):                                  # TODO: Test this functionality

                # Construct metric
                with DumbCheckpoint(di + 'hdf5/Error2d_' + indexString(int(cnt / op.rm)),
                                    mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(interp(mesh_H, epsilon))
                M = isotropicMetric(errEst, invert=False, op=op)        # TODO: Not sure normalisation is working
                if op.gradate:
                    M_ = isotropicMetric(interp(mesh_H, H0), bdy=True, op=op)  # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=True)
                    metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                P1 = FunctionSpace(mesh_H, "CG", 1)
                elev_2d, uv_2d = interp(mesh_H, elev_2d, uv_2d)
                b = problemDomain(mesh=mesh_H, op=op)[3]  # Reset bathymetry on new mesh
                uv_2d.rename('uv_2d')
                elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh_H, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.use_grad_depth_viscosity_term = False               # TODO: Might as well include this for Tohoku?
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = di
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
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
            cb2 = SWCallback(adapSolver)
            cb2.op = op
            if op.mode == 'tohoku':
                cb3 = P02Callback(adapSolver)
                cb4 = P06Callback(adapSolver)
                if cnt == 0:
                    initP02 = cb2.init_value
                    initP06 = cb3.init_value
            if cnt != 0:
                cb2.objective_value = integrand
                if op.mode == 'tohoku':
                    cb3.gauge_values = gP02
                    cb3.init_value = initP02
                    cb4.gauge_values = gP06
                    cb4.init_value = initP06
            adapSolver.add_callback(cb2, 'timestep')
            if op.mode == 'tohoku':
                adapSolver.add_callback(cb3, 'timestep')
                adapSolver.add_callback(cb4, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            J_h = cb2.quadrature()
            integrand = cb2.__call__()[1]
            if op.mode == 'tohoku':
                gP02 = cb3.__call__()[1]
                gP06 = cb4.__call__()[1]

            # Get mesh stats
            nEle = meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)

            adaptSolveTimer += adaptTimer + solverTimer

        if op.mode == 'tohoku':
            totalVarP02 = cb3.totalVariation()
            totalVarP06 = cb4.totalVariation()

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT / op.ndump)
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return av, rel, J_h, integrand, np.abs(
                peak / peak_a), distance, distance / distance_a, adaptSolveTimer
        elif op.mode == 'tohoku':
            return av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, adaptSolveTimer
        else:
            return av, rel, J_h, integrand, adaptSolveTimer


if __name__ == "__main__":
    import argparse
    import datetime


    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'hessianBased', 'DWP', 'DWR'} (default 'fixedMesh')")
    parser.add_argument("-t", help="Choose test problem from {'shallow-water', 'rossby-wave'} (default 'tohoku')")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-ho", help="Compute errors and residuals in a higher order space")
    parser.add_argument("-lo", help="Compute errors and residuals in a lower order space")
    parser.add_argument("-r", help="Compute errors and residuals in a refined space")
    parser.add_argument("-b", help="Intersect metrics with bathymetry")
    parser.add_argument("-regen", help="Regenerate error estimates based on saved data")
    args = parser.parse_args()

    solvers = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased, 'DWP': DWP, 'DWR': DWR}
    approach = args.a
    if approach is None:
        approach = 'fixedMesh'
    else:
        assert approach in solvers.keys()
    solver = solvers[approach]
    if args.t is None:
        mode = 'tohoku'
    else:
        mode = args.t
    print("Mode: %s, approach: %s" % (mode, approach))
    orderChange = 0
    if args.ho:
        assert (not args.r) and (not args.lo)
        orderChange = 1
    if args.lo:
        assert (not args.r) and (not args.ho)
        orderChange = -1
    if args.r:
        assert (not args.ho) and (not args.lo)
    if args.b is not None:
        assert approach == 'hessianBased'

    # Choose mode and set parameter values
    op = Options(mode=mode,
                 approach=approach,
                 # gradate=True if approach in ('DWP', 'DWR') and mode == 'tohoku' else False,
                 gradate=False,  # TODO: Fix this for tohoku case
                 plotpvd=True if args.o else False,
                 orderChange=orderChange,
                 refinedSpace=True if args.r else False,
                 bAdapt=bool(args.b) if args.b is not None else False,
                 regen=bool(args.regen) if args.regen is not None else False)

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
    if args.b:
        filename += '_b'
    filename += '_' + date
    errorfile = open(filename + '.txt', 'w+')
    integrandFile = open(filename + 'Integrand.txt', 'w+')

    # Run simulations
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    Jlist = np.zeros(len(resolutions))
    if mode == 'tohoku':
        gaugeFileP02 = open(filename + 'P02.txt', 'w+')
        gaugeFileP06 = open(filename + 'P06.txt', 'w+')
    for i in resolutions:
        # Get data and save to disk
        if mode == 'rossby-wave':   # TODO: Timing output does not work in adjoint cases
            av, rel, J_h, integrand, relativePeak, distance, phaseSpd, solverTime = solver(i, op=op)
            print("""Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs
        OF error: %.4e  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm"""
                  % (i, av, J_h, solverTime, rel, relativePeak, distance, phaseSpd))
            errorfile.write('%d, %.4e, %.4f, %.4f, %.4f, %.1f, %.4e\n'
                           % (av, rel, relativePeak, distance, phaseSpd, solverTime, J_h))
        elif mode == 'tohoku':
            av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, solverTime = solver(i, op=op)
            print("""Run %d: Mean element count: %6d Objective %.4e Timing %.1fs 
        OF error: %.4e P02: %.3f P06: %.3f""" % (i, av, J_h, solverTime, rel, totalVarP02, totalVarP06))
            errorfile.write('%d, %.4e, %.3f, %.3f, %.1f, %.4e\n'
                            % (av, rel, totalVarP02, totalVarP06, solverTime, J_h))
            gaugeFileP02.writelines(["%s," % val for val in gP02])
            gaugeFileP02.write("\n")
            gaugeFileP06.writelines(["%s," % val for val in gP06])
            gaugeFileP06.write("\n")
        else:
            av, rel, J_h, integrand, solverTime = solver(i, op=op)
            print('Run %d: Mean element count: %6d Objective: %.4e OF error %.4e Timing %.1fs'
                  % (i, av, J_h, rel, solverTime))
            errorfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, solverTime, J_h))
        integrandFile.writelines(["%s," % val for val in integrand])
        integrandFile.write("\n")
    integrandFile.close()
