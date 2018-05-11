from thetis_adjoint import *
from thetis.field_defs import field_metadata
import pyadjoint
from fenics_adjoint.solving import SolveBlock                                       # For extracting adjoint solutions

import numpy as np
from time import clock

from utils.adaptivity import *
from utils.callbacks import *
from utils.interpolation import interp, mixedPairInterp
from utils.setup import problemDomain, RossbyWaveSolution
from utils.misc import indexString, peakAndDistance, meshStats
from utils.options import Options


def fixedMesh(startRes, **kwargs):
    op = kwargs.get('op')

    with pyadjoint.stop_annotating():

        # Initialise domain and physical parameters
        try:
            assert float(physical_constants['g_grav'].dat.data) == op.g
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        nEle = meshStats(mesh)[0]
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()    # Needed to load data into
        if op.mode == 'rossby-wave':            # Analytic final-time state
            peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])

        # Initialise solver
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.use_grad_div_viscosity_term = True              # Symmetric viscous stress
        options.use_lax_friedrichs_velocity = False             # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.ndump
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.output_directory = op.di
        options.no_exports = True
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = SWCallback(solver_obj)
        cb1.op = op
        if op.mode != 'tohoku':
            cb2 = MirroredSWCallback(solver_obj)
            cb2.op = op
            solver_obj.add_callback(cb2, 'timestep')
        else:
            cb3 = P02Callback(solver_obj)
            cb4 = P06Callback(solver_obj)
            solver_obj.add_callback(cb3, 'timestep')
            solver_obj.add_callback(cb4, 'timestep')
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs

        # Solve and extract timeseries / functionals
        quantities = {}
        solverTimer = clock()
        solver_obj.iterate()
        solverTimer = clock() - solverTimer
        quantities['J_h'] = cb1.quadrature()          # Evaluate objective functional
        quantities['Integrand'] = cb1.getVals()
        if op.mode != 'tohoku':
            quantities['J_h mirrored'] = cb2.quadrature()
            quantities['Integrand-mirrored'] = cb2.getVals()
        else:
            quantities['TV P02'] = cb3.totalVariation()
            quantities['TV P06'] = cb4.totalVariation()
            quantities['P02'] = cb3.getVals()
            quantities['P06'] = cb4.getVals()

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            peak, distance = peakAndDistance(solver_obj.fields.solution_2d.split()[1], op=op)
            quantities['peak'] = peak/peak_a
            quantities['dist'] = distance/distance_a
            quantities['spd'] = distance /(op.Tend * 0.4)

        # Output mesh statistics and solver times
        quantities['meanElements'] = nEle
        quantities['solverTimer'] = solverTimer
        quantities['adaptSolveTimer'] = 0.

        return quantities


def hessianBased(startRes, **kwargs):
    op = kwargs.get('op')

    with pyadjoint.stop_annotating():

        # Initialise domain and physical parameters
        try:
            assert float(physical_constants['g_grav'].dat.data) == op.g
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        if op.mode == 'rossby-wave':    # Analytic final-time state
            peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])

        # Initialise parameters and counters
        nEle, op.nVerT = meshStats(mesh)
        op.nVerT *= op.rescaling  # Target #Vertices
        mM = [nEle, nEle]  # Min/max #Elements
        Sn = nEle
        cnt = 0
        endT = 0.

        adaptSolveTimer = 0.
        quantities = {}
        while cnt < op.cntT:
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
                    if cnt != 0:
                        uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
                    elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
                    b, BCs, f = problemDomain(mesh=mesh, op=op)[3:]
                    uv_2d.rename('uv_2d')
                    elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = op.di
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.coriolis_frequency = f
            field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
            e = exporter.ExportManager(op.di + 'hdf5',
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
            if op.mode != 'tohoku':
                cb2 = MirroredSWCallback(adapSolver)
                cb2.op = op
            else:
                cb3 = P02Callback(adapSolver)
                cb4 = P06Callback(adapSolver)
                if cnt == 0:
                    initP02 = cb3.init_value
                    initP06 = cb4.init_value
            if cnt != 0:
                cb1.objective_value = quantities['Integrand']
                if op.mode != 'tohoku':
                    cb2.objective_value = quantities['Integrand-mirrored']
                else:
                    cb3.gauge_values = quantities['P02']
                    cb3.init_value = initP02
                    cb4.gauge_values = quantities['P06']
                    cb4.init_value = initP06
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode != 'tohoku':
                adapSolver.add_callback(cb2, 'timestep')
            else:
                adapSolver.add_callback(cb3, 'timestep')
                adapSolver.add_callback(cb4, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
            quantities['Integrand'] = cb1.getVals()
            if op.mode != 'tohoku':
                quantities['J_h mirrored'] = cb2.quadrature()
                quantities['Integrand-mirrored'] = cb2.getVals()
            else:
                quantities['P02'] = cb3.getVals()
                quantities['P06'] = cb4.getVals()
                quantities['TV P02'] = cb3.totalVariation()
                quantities['TV P06'] = cb4.totalVariation()

            # Get mesh stats
            nEle = meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt/op.rm+1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
            adaptSolveTimer += adaptTimer + solverTimer

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            peak, distance = peakAndDistance(adapSolver.fields.solution_2d.split()[1], op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.Tend * 0.4)

        # Output mesh statistics and solver times
        quantities['meanElements'] = av
        quantities['solverTimer'] = adaptSolveTimer
        quantities['adaptSolveTimer'] = adaptSolveTimer

        return quantities


def DWP(startRes, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')

    initTimer = clock()
    if op.plotpvd:
        errorFile = File(op.di + "ErrorIndicator2d.pvd")
        adjointFile = File(op.di + "Adjoint2d.pvd")

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
    V = op.mixedSpace(mesh)
    q = Function(V)
    uv_2d, elev_2d = q.split()  # Needed to load data into
    uv_2d.rename('uv_2d')
    elev_2d.rename('elev_2d')
    P1 = FunctionSpace(mesh, "CG", 1)
    if op.mode == 'rossby-wave':    # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
    epsilon = Function(P1, name="Error indicator")
    epsilon_ = Function(P1)

    # Initialise parameters and counters
    nEle, op.nVerT = meshStats(mesh)
    op.nVerT *= op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    endT = 0.

    # Get initial boundary metric
    if op.gradate:
        H0 = Function(P1).interpolate(CellSize(mesh))

    if not regen:

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.use_grad_div_viscosity_term = True                      # Symmetric viscous stress
        options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.rm
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.output_directory = op.di
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        cb1.mirror = kwargs.get('mirror')
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)
        primalTimer = clock()
        solver_obj.iterate()
        primalTimer = clock() - primalTimer
        J = cb1.quadrature()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        dJdb = compute_gradient(J, Control(b))
        gradientTimer = clock() - gradientTimer
        print("Norm of gradient: %.3e. Computation time: %.1fs" % (dJdb.dat.norm, gradientTimer))

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.ndump                            # Number of extra tape annotations in setup
        for i in range(N - 1, r - 2, -op.ndump):
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            with DumbCheckpoint(op.di + 'hdf5/Adjoint2d_' + indexString(int((i - r + 1) / op.ndump)), mode=FILE_CREATE) as saveAdj:
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
        for k in range(0, op.rmEnd):  # Loop back over times to generate error estimators
            print('Generating error estimate %d / %d' % (k + 1, op.rmEnd))
            with DumbCheckpoint(op.di + 'hdf5/Velocity2d_' + indexString(k), mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d)
                loadVel.close()
            with DumbCheckpoint(op.di + 'hdf5/Elevation2d_' + indexString(k), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d)
                loadElev.close()

            # Load adjoint data and form indicators
            epsilon.interpolate(inner(q, dual))
            for i in range(k, min(k + op.iEnd - op.iStart, op.iEnd)):
                with DumbCheckpoint(op.di + 'hdf5/Adjoint2d_' + indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_u)
                    loadAdj.load(dual_e)
                    loadAdj.close()
                epsilon_.interpolate(inner(q, dual))
                epsilon = pointwiseMax(epsilon, epsilon_)
            epsilon = normaliseIndicator(epsilon, op=op)
            with DumbCheckpoint(op.di + 'hdf5/ErrorIndicator2d_' + indexString(k), mode=FILE_CREATE) as saveErr:
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
        quantities = {}
        while cnt < op.cntT:
            adaptTimer = clock()
            for l in range(op.nAdapt):                                  # TODO: Test this functionality

                # Construct metric
                with DumbCheckpoint(op.di + 'hdf5/ErrorIndicator2d_' + indexString(int(cnt/op.rm)), mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(interp(mesh, epsilon))
                M = isotropicMetric(errEst, invert=False, op=op)
                if op.gradate:
                    M_ = isotropicMetric(interp(mesh, H0), bdy=True, op=op)  # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=True)
                    metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                if cnt != 0:
                    uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
                elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
                b, BCs, f = problemDomain(mesh=mesh, op=op)[3:]
                uv_2d.rename('uv_2d')
                elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = op.di
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.coriolis_frequency = f
            field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
            e = exporter.ExportManager(op.di + 'hdf5',
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
            cb1 = SWCallback(adapSolver)
            cb1.op = op
            if op.mode != 'tohoku':
                cb2 = MirroredSWCallback(adapSolver)
                cb2.op = op
            else:
                cb3 = P02Callback(adapSolver)
                cb4 = P06Callback(adapSolver)
                if cnt == 0:
                    initP02 = cb3.init_value
                    initP06 = cb4.init_value
            if cnt != 0:
                cb1.objective_value = quantities['Integrand']
                if op.mode != 'tohoku':
                    cb2.objective_value = quantities['Integrand-mirrored']
                else:
                    cb3.gauge_values = quantities['P02']
                    cb3.init_value = initP02
                    cb4.gauge_values = quantities['P06']
                    cb4.init_value = initP06
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode != 'tohoku':
                adapSolver.add_callback(cb2, 'timestep')
            else:
                adapSolver.add_callback(cb3, 'timestep')
                adapSolver.add_callback(cb4, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
            quantities['Integrand'] = cb1.getVals()
            if op.mode != 'tohoku':
                quantities['J_h mirrored'] = cb2.quadrature()
                quantities['Integrand-mirrored'] = cb2.getVals()
            else:
                quantities['P02'] = cb3.getVals()
                quantities['P06'] = cb4.getVals()
                quantities['TV P02'] = cb3.totalVariation()
                quantities['TV P06'] = cb4.totalVariation()

            # Get mesh stats
            nEle = meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
            adaptSolveTimer += adaptTimer + solverTimer

            # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            peak, distance = peakAndDistance(adapSolver.fields.solution_2d.split()[1], op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.Tend * 0.4)

        # Output mesh statistics and solver times
        totalTimer = errorTimer + adaptSolveTimer
        if not regen:
            totalTimer += primalTimer + gradientTimer + dualTimer
        quantities['meanElements'] = av
        quantities['solverTimer'] = totalTimer
        quantities['adaptSolveTimer'] = adaptSolveTimer

        return quantities


def DWR(startRes, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')

    initTimer = clock()
    if op.plotpvd:
        residualFile = File(op.di + "Residual2d.pvd")
        errorFile = File(op.di + "ErrorIndicator2d.pvd")
        adjointFile = File(op.di + "Adjoint2d.pvd")

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
    if op.mode == 'rossby-wave':    # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")

    if op.orderChange:
        Ve = op.mixedSpace(mesh_H, enrich=True)
        duale = Function(Ve)
        duale_u, duale_e = duale.split()
        epsilon = Function(P1, name="Error indicator")
    elif op.refinedSpace:                   # Define variables on an iso-P2 refined space
        mesh_h = isoP2(mesh_H)
        Ve = op.mixedSpace(mesh_h)
        epsilon = Function(FunctionSpace(mesh_h, "CG", 1), name="Error indicator")
    else:                                   # Copy standard variables to mimic enriched space labels
        Ve = V
        epsilon = Function(P1, name="Error indicator")
    v = TestFunction(FunctionSpace(mesh_h if op.refinedSpace else mesh_H, "DG", 0)) # For forming error indicators
    rho = Function(Ve)
    rho_u, rho_e = rho.split()
    rho_u.rename("Momentum error")
    rho_e.rename("Continuity error")

    # Initialise parameters and counters
    nEle, op.nVerT = meshStats(mesh_H)
    op.nVerT *= op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    endT = 0.

    # Get initial boundary metric
    if op.gradate:
        H0 = Function(P1).interpolate(CellSize(mesh_H))

    if not regen:

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh_H, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.use_grad_div_viscosity_term = True                      # Symmetric viscous stress
        options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.rm
        options.simulation_end_time = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.output_directory = op.di   # Need this for residual callback
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']            # TODO: EXPORT FROM PREVIOUS STEP?
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        cb1.mirror = kwargs.get('mirror')
        # if op.orderChange:
        #     cb2 = HigherOrderResidualCallback(solver_obj, Ve)
        # elif op.refinedSpace:
        #     cb2 = RefinedResidualCallback(solver_obj, Ve)
        # else:
        #     cb2 = ResidualCallback(solver_obj)
        solver_obj.add_callback(cb1, 'timestep')
        # solver_obj.add_callback(cb2, 'export')
        solver_obj.bnd_functions['shallow_water'] = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)

        # primalTimer = clock()
        # solver_obj.iterate()
        # primalTimer = clock() - primalTimer

        cnt = 0
        primalTimer = 0.
        options.simulation_end_time = op.dt * op.rm
        while solver_obj.simulation_time < op.Tend - 0.5 * op.dt:

            # # Calculate and store residuals
            # with pyadjoint.stop_annotating():
            #     err_u, err_e = strongResidualSW(solver_obj, Ve, op=op)
            #     rho_u.interpolate(err_u)
            #     rho_e.interpolate(err_e)
            #     with DumbCheckpoint(op.di + 'hdf5/Error2d_' + indexString(cnt), mode=FILE_CREATE) as saveRes:
            #         saveRes.store(rho_u)
            #         saveRes.store(rho_e)
            #         saveRes.close()
            #     if op.plotpvd:
            #         residualFile.write(rho_u, rho_e, time=solver_obj.simulation_time)

            with pyadjoint.stop_annotating():
                uv_old, elev_old = solver_obj.timestepper.solution_old.split()
                uv_old.rename("Previous velocity")
                elev_old.rename("Previous elevation")
                with DumbCheckpoint(op.di + 'hdf5/Previous2d_' + indexString(cnt), mode=FILE_CREATE) as savePrev:
                    savePrev.store(uv_old)
                    savePrev.store(elev_old)
                    savePrev.close()

                if cnt != 0:
                    solver_obj.load_state(cnt, iteration=cnt*op.rm)

            # Run simulation
            stepTimer = clock()
            solver_obj.iterate()
            stepTimer = clock() - stepTimer
            primalTimer += stepTimer
            options.simulation_end_time += op.dt * op.rm
            cnt += 1

        J = cb1.quadrature()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        dJdb = compute_gradient(J, Control(b))
        gradientTimer = clock() - gradientTimer

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.rm   # Number of extra tape annotations in setup
        for i in range(N - 1, r - 2, -op.rm):
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            with DumbCheckpoint(op.di + 'hdf5/Adjoint2d_' + indexString(int((i - r + 1) / op.rm)),  mode=FILE_CREATE) as saveAdj:
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
        for k in range(0, int(op.cntT / op.rm)):
            print('Generating error estimate %d / %d' % (k + 1, int(op.cntT / op.rm)))
            # with DumbCheckpoint(op.di + 'hdf5/Error2d_' + indexString(k), mode=FILE_READ) as loadRes:
            #     loadRes.load(rho_u, name="Momentum error")
            #     loadRes.load(rho_e, name="Continuity error")
            #     loadRes.close()

            # Generate residuals
            with DumbCheckpoint(op.di + 'hdf5/Velocity2d_' + indexString(k), mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d, name="uv_2d")
                loadVel.close()
            with DumbCheckpoint(op.di + 'hdf5/Elevation2d_' + indexString(k), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name="elev_2d")
                loadElev.close()
            with DumbCheckpoint(op.di + 'hdf5/Previous2d_' + indexString(k), mode=FILE_READ) as loadPrev:
                loadPrev.load(uv_old, name="Previous velocity")
                loadPrev.load(elev_old, name="Previous elevation")
                loadPrev.close()
            err_u, err_e = strongResidualSW(solver_obj, uv_2d, elev_2d, uv_old, elev_old, Ve, op=op)
            rho_u.interpolate(err_u)
            rho_e.interpolate(err_e)
            if op.plotpvd:
                residualFile.write(rho_u, rho_e, time=float(op.dt * op.rm * k))


            # Load adjoint data and form indicators
            with DumbCheckpoint(op.di + 'hdf5/Adjoint2d_' + indexString(k), mode=FILE_READ) as loadAdj:
                loadAdj.load(dual_u)
                loadAdj.load(dual_e)
                loadAdj.close()
            if op.orderChange:
                duale_u.interpolate(dual_u)
                duale_e.interpolate(dual_e)
                epsilon.interpolate(assemble(v * inner(rho, duale) * dx))
            elif op.refinedSpace:
                duale = mixedPairInterp(mesh_h, dual)
                epsilon.interpolate(assemble(v * inner(rho, duale) * dx))
            else:
                epsilon.interpolate(assemble(v * inner(rho, dual) * dx))
            epsilon = normaliseIndicator(epsilon, op=op)
            epsilon.rename("Error indicator")   # TODO: Try scaling by H0 as in Rannacher 08
            with DumbCheckpoint(op.di + 'hdf5/ErrorIndicator2d_' + indexString(k), mode=FILE_CREATE) as saveErr:
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
        quantities = {}
        while cnt < op.cntT:
            adaptTimer = clock()
            for l in range(op.nAdapt):                                          # TODO: Test this functionality

                # Construct metric
                indexStr = indexString(int(cnt / op.rm))
                with DumbCheckpoint(op.di + 'hdf5/ErrorIndicator2d_' + indexStr, mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh_H, "CG", 1)).assign(interp(mesh_H, epsilon))
                M = isotropicMetric(errEst, invert=False, op=op)
                if op.gradate:
                    M_ = isotropicMetric(interp(mesh_H, H0), bdy=True, op=op)   # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=True)
                    metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                if cnt != 0:
                    uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
                elev_2d, uv_2d = interp(mesh_H, elev_2d, uv_2d)
                b, BCs, f = problemDomain(mesh=mesh_H, op=op)[3:]
                uv_2d.rename('uv_2d')
                elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh_H, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = op.di
            adapOpt.export_diagnostics = True
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.coriolis_frequency = f
            e = exporter.ExportManager(op.di + 'hdf5',
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
            cb1 = SWCallback(adapSolver)
            cb1.op = op
            if op.mode != 'tohoku':
                cb2 = MirroredSWCallback(adapSolver)
                cb2.op = op
            else:
                cb3 = P02Callback(adapSolver)
                cb4 = P06Callback(adapSolver)
                if cnt == 0:
                    initP02 = cb3.init_value
                    initP06 = cb4.init_value
            if cnt != 0:
                cb1.objective_value = quantities['Integrand']
                if op.mode != 'tohoku':
                    cb2.objective_value = quantities['Integrand-mirrored']
                else:
                    cb3.gauge_values = quantities['P02']
                    cb3.init_value = initP02
                    cb4.gauge_values = quantities['P06']
                    cb4.init_value = initP06
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode != 'tohoku':
                adapSolver.add_callback(cb2, 'timestep')
            else:
                adapSolver.add_callback(cb3, 'timestep')
                adapSolver.add_callback(cb4, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
            quantities['Integrand'] = cb1.getVals()
            if op.mode != 'tohoku':
                quantities['J_h mirrored'] = cb2.quadrature()
                quantities['Integrand-mirrored'] = cb2.getVals()
            else:
                quantities['P02'] = cb3.getVals()
                quantities['P06'] = cb4.getVals()
                quantities['TV P02'] = cb3.totalVariation()
                quantities['TV P06'] = cb4.totalVariation()

            # Get mesh stats
            nEle = meshStats(mesh_H)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
            adaptSolveTimer += adaptTimer + solverTimer

            # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            peak, distance = peakAndDistance(adapSolver.fields.solution_2d.split()[1], op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.Tend * 0.4)

            # Output mesh statistics and solver times
        totalTimer = errorTimer + adaptSolveTimer
        if not regen:
            totalTimer += primalTimer + gradientTimer + dualTimer
        quantities['meanElements'] = av
        quantities['solverTimer'] = totalTimer
        quantities['adaptSolveTimer'] = adaptSolveTimer

        return quantities


# def DWR(startRes, **kwargs):
#     op = kwargs.get('op')
#     regen = kwargs.get('regen')
#
#     initTimer = clock()
#     if op.plotpvd:
#         residualFile = File(op.di + "Residual2d.pvd")
#         errorFile = File(op.di + "ErrorIndicator2d.pvd")
#         adjointFile = File(op.di + "Adjoint2d.pvd")
#
#     # Initialise domain and physical parameters
#     try:
#         assert (float(physical_constants['g_grav'].dat.data) == op.g)
#     except:
#         physical_constants['g_grav'].assign(op.g)
#     mesh_H, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
#     V = op.mixedSpace(mesh_H)
#     q = Function(V)
#     uv_2d, elev_2d = q.split()    # Needed to load data into
#     uv_2d.rename('uv_2d')
#     elev_2d.rename('elev_2d')
#     P1 = FunctionSpace(mesh_H, "CG", 1)
#     if op.mode == 'rossby-wave':    # Analytic final-time state
#         peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.Tend).split()[1])
#
#     # Define Functions relating to a posteriori DWR error estimator
#     dual = Function(V)
#     dual_u, dual_e = dual.split()
#     dual_u.rename("Adjoint velocity")
#     dual_e.rename("Adjoint elevation")
#     new_dual = Function(V)
#     new_dual_u, new_dual_e = new_dual.split()
#
#     # Initialise parameters and counters
#     nEle, op.nVerT = meshStats(mesh_H)
#     op.nVerT *= op.rescaling  # Target #Vertices
#     mM = [nEle, nEle]  # Min/max #Elements
#     Sn = nEle
#     endT = 0.
#
#     # Get initial boundary metric
#     if op.gradate:
#         H0 = Function(P1).interpolate(CellSize(mesh_H))
#
#     if not regen:
#
#         # Solve fixed mesh primal problem to get residuals and adjoint solutions
#         solver_obj = solver2d.FlowSolver2d(mesh_H, b)
#         options = solver_obj.options
#         options.element_family = op.family
#         options.use_nonlinear_equations = True
#         options.use_grad_div_viscosity_term = True                      # Symmetric viscous stress
#         options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
#         options.coriolis_frequency = f
#         options.simulation_export_time = op.dt * op.rm
#         options.simulation_end_time = op.Tend
#         options.timestepper_type = op.timestepper
#         options.timestep = op.dt
#         options.output_directory = op.di
#         options.no_exports = True
#         solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
#         cb1 = ObjectiveSWCallback(solver_obj)
#         cb1.op = op
#         cb1.mirror = kwargs.get('mirror')
#         solver_obj.add_callback(cb1, 'timestep')
#         solver_obj.bnd_functions['shallow_water'] = BCs
#         initTimer = clock() - initTimer
#         print('Problem initialised. Setup time: %.3fs' % initTimer)
#         primalTimer = clock()
#         solver_obj.iterate()
#         primalTimer = clock() - primalTimer
#         J = cb1.quadrature()                        # Assemble objective functional for adjoint computation
#         print('Primal run complete. Solver time: %.3fs' % primalTimer)
#
#         # Compute gradient
#         gradientTimer = clock()
#         dJdb = compute_gradient(J, Control(b))
#         gradientTimer = clock() - gradientTimer
#         print("Norm of gradient: %.3e. Computation time: %.1fs" % (dJdb.dat.norm, gradientTimer))
#
#         # Extract adjoint solutions
#         dualTimer = clock()
#         tape = get_working_tape()
#         solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
#         N = len(solve_blocks)
#         r = N % op.rm   # Number of extra tape annotations in setup
#         for i in range(N - 1, r - 2, -op.rm):
#             dual.assign(solve_blocks[i].adj_sol)
#             dual_u, dual_e = dual.split()
#             with DumbCheckpoint(op.di + 'hdf5/Adjoint2d_' + indexString(int((i - r + 1) / op.rm)),  mode=FILE_CREATE) as saveAdj:
#                 saveAdj.store(dual_u)
#                 saveAdj.store(dual_e)
#                 saveAdj.close()
#             if op.plotpvd:
#                 adjointFile.write(dual_u, dual_e, time=op.dt * (i - r + 1))
#             print('Adjoint simulation %.2f%% complete' % ((N - i + r - 1) / N * 100))
#         dualTimer = clock() - dualTimer
#         print('Dual run complete. Run time: %.3fs' % dualTimer)
#
#     # Adaptive primal run
#     with pyadjoint.stop_annotating():
#         cnt = 0
#         adaptSolveTimer = 0.
#         q = Function(V)
#         uv_2d, elev_2d = q.split()
#         elev_2d.interpolate(eta0)
#         uv_2d.interpolate(u0)
#         quantities = {}
#         while cnt < op.cntT:
#             adaptTimer = clock()
#             V = op.mixedSpace(mesh_H)
#             if op.orderChange:              # Define variables on higher/lower order space
#                 Ve = op.mixedSpace(mesh_H)  # Automatically generates a higher/lower order space
#                 epsilon = Function(P1, name="Error indicator")
#             elif op.refinedSpace:           # Define variables on an iso-P2 refined space
#                 mesh_h = isoP2(mesh_H)
#                 Ve = op.mixedSpace(mesh_h)
#                 epsilon = Function(FunctionSpace(mesh_h, "CG", 1), name="Error indicator")
#             else:                           # Copy standard variables to mimic enriched space labels
#                 Ve = V
#                 epsilon = Function(P1, name="Error indicator")
#             duale = Function(Ve)
#             duale_u, duale_e = duale.split()
#             v = TestFunction(FunctionSpace(mesh_h if op.refinedSpace else mesh_H, "DG", 0))
#             residual = Function(Ve)
#             residual_u, residual_e = residual.split()
#             if cnt == 0:                    # For initial residual
#                 adapSolver = solver2d.FlowSolver2d(mesh_H, b)
#                 adapOpt = adapSolver.options
#                 adapOpt.element_family = op.family
#                 adapOpt.use_nonlinear_equations = True
#                 adapOpt.use_grad_div_viscosity_term = True   # Symmetric viscous stress
#                 adapOpt.use_lax_friedrichs_velocity = False  # TODO: This is a temporary fix
#                 adapOpt.timestep = op.dt
#                 adapOpt.no_exports = True
#                 adapOpt.coriolis_frequency = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(
#                     SpatialCoordinate(mesh_H)[1])
#                 adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
#             rho_u, rho_e = strongResidualSW(adapSolver)
#             residual_u.interpolate(rho_u)
#             residual_e.interpolate(rho_e)
#             residual_u.rename("Velocity residual")
#             residual_e.rename("Elevation residual")
#             if op.plotpvd:
#                 residualFile.write(residual_u, residual_e, time=float(op.dt * cnt))
#
#             # Load adjoint data and form indicators
#             with DumbCheckpoint(op.di + 'hdf5/Adjoint2d_' + indexString(int(cnt/op.rm)), mode=FILE_READ) as loadAdj:
#                 loadAdj.load(dual_u)
#                 loadAdj.load(dual_e)
#                 loadAdj.close()
#             if cnt != 0:
#                 new_dual_u, new_dual_e = interp(mesh_H, dual_u, dual_e)
#             else:
#                 new_dual_u.interpolate(dual_u)
#                 new_dual_e.interpolate(dual_e)
#             if op.refinedSpace:
#                 raise NotImplementedError
#             duale_u.interpolate(new_dual_u)
#             duale_e.interpolate(new_dual_e)
#             epsilon.interpolate(assemble(v * inner(residual, duale) * dx))
#             epsilon = normaliseIndicator(epsilon, op=op)
#             epsilon.rename("Error indicator")   # TODO: Try scaling by H0 as in Rannacher 08
#             if op.plotpvd:
#                 errorFile.write(epsilon, time=float(cnt/op.rm))
#
#             for l in range(op.nAdapt):          # TODO: Test this functionality
#
#                 # Construct metric
#                 M = isotropicMetric(epsilon, invert=False, op=op)
#                 if op.gradate:
#                     M_ = isotropicMetric(interp(mesh_H, H0), bdy=True, op=op)   # Initial boundary metric
#                     M = metricIntersection(M, M_, bdy=True)
#                     metricGradation(M, op=op)
#
#                 # Adapt mesh and interpolate variables
#                 mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
#                 if cnt != 0:
#                     uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
#                 elev_2d, uv_2d = interp(mesh_H, elev_2d, uv_2d)
#                 P1 = FunctionSpace(mesh_H, "CG", 1)
#                 b, BCs, f = problemDomain(mesh=mesh_H, op=op)[3:]
#                 uv_2d.rename('uv_2d')
#                 elev_2d.rename('elev_2d')
#             adaptTimer = clock() - adaptTimer
#
#             # Solver object and equations
#             adapSolver = solver2d.FlowSolver2d(mesh_H, b)
#             adapOpt = adapSolver.options
#             adapOpt.element_family = op.family
#             adapOpt.use_nonlinear_equations = True
#             adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
#             adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
#             adapOpt.simulation_export_time = op.dt * op.ndump
#             startT = endT
#             endT += op.dt * op.rm
#             adapOpt.simulation_end_time = endT
#             adapOpt.timestepper_type = op.timestepper
#             adapOpt.timestep = op.dt
#             adapOpt.output_directory = op.di
#             adapOpt.export_diagnostics = True
#             adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
#             adapOpt.coriolis_frequency = f
#             e = exporter.ExportManager(op.di + 'hdf5',
#                                        ['elev_2d', 'uv_2d'],
#                                        {'elev_2d': elev_2d, 'uv_2d': uv_2d},
#                                        field_metadata,
#                                        export_type='hdf5')
#             adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
#             adapSolver.i_export = int(cnt / op.ndump)
#             adapSolver.iteration = cnt
#             adapSolver.simulation_time = startT
#             adapSolver.next_export_t = startT + adapOpt.simulation_export_time  # For next export
#             for e in adapSolver.exporters.values():
#                 e.set_next_export_ix(adapSolver.i_export)
#
#             # Evaluate callbacks and iterate
#             cb1 = SWCallback(adapSolver)
#             cb1.op = op
#             if op.mode != 'tohoku':
#                 cb2 = MirroredSWCallback(adapSolver)
#                 cb2.op = op
#             else:
#                 cb3 = P02Callback(adapSolver)
#                 cb4 = P06Callback(adapSolver)
#                 if cnt == 0:
#                     initP02 = cb3.init_value
#                     initP06 = cb4.init_value
#             if cnt != 0:
#                 cb1.objective_value = quantities['Integrand']
#                 if op.mode != 'tohoku':
#                     cb2.objective_value = quantities['Integrand-mirrored']
#                 else:
#                     cb3.gauge_values = quantities['P02']
#                     cb3.init_value = initP02
#                     cb4.gauge_values = quantities['P06']
#                     cb4.init_value = initP06
#             adapSolver.add_callback(cb1, 'timestep')
#             if op.mode != 'tohoku':
#                 adapSolver.add_callback(cb2, 'timestep')
#             else:
#                 adapSolver.add_callback(cb3, 'timestep')
#                 adapSolver.add_callback(cb4, 'timestep')
#             adapSolver.bnd_functions['shallow_water'] = BCs
#             solverTimer = clock()
#             adapSolver.iterate()
#             solverTimer = clock() - solverTimer
#             quantities['J_h'] = cb1.quadrature()  # Evaluate objective functional
#             quantities['Integrand'] = cb1.getVals()
#             if op.mode != 'tohoku':
#                 quantities['J_h mirrored'] = cb2.quadrature()
#                 quantities['Integrand-mirrored'] = cb2.getVals()
#             else:
#                 quantities['P02'] = cb3.getVals()
#                 quantities['P06'] = cb4.getVals()
#                 quantities['TV P02'] = cb3.totalVariation()
#                 quantities['TV P06'] = cb4.totalVariation()
#
#             # Get mesh stats
#             nEle = meshStats(mesh_H)[0]
#             mM = [min(nEle, mM[0]), max(nEle, mM[1])]
#             Sn += nEle
#             cnt += op.rm
#             av = op.printToScreen(int(cnt / op.rm + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
#             adaptSolveTimer += adaptTimer + solverTimer
#
#             # Measure error using metrics, as in Huang et al.
#         if op.mode == 'rossby-wave':
#             peak, distance = peakAndDistance(adapSolver.fields.solution_2d.split()[1], op=op)
#             quantities['peak'] = peak / peak_a
#             quantities['dist'] = distance / distance_a
#             quantities['spd'] = distance / (op.Tend * 0.4)
#
#         # Output mesh statistics and solver times
#         totalTimer = adaptSolveTimer
#         if not regen:
#             totalTimer += primalTimer + gradientTimer + dualTimer
#         quantities['meanElements'] = av
#         quantities['solverTimer'] = totalTimer
#         quantities['adaptSolveTimer'] = adaptSolveTimer
#
#         return quantities


if __name__ == "__main__":
    import argparse
    import datetime


    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("t", help="Choose test problem from {'shallow-water', 'rossby-wave'} (default 'tohoku')")
    parser.add_argument("-a", help="Choose adaptive approach from {'hessianBased', 'DWP', 'DWR'} (default 'fixedMesh')")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    parser.add_argument("-ho", help="Compute errors and residuals in a higher order space")
    parser.add_argument("-r", help="Compute errors and residuals in a refined space")
    parser.add_argument("-b", help="Intersect metrics with bathymetry")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-regen", help="Regenerate error estimates from saved data")
    parser.add_argument("-mirror", help="Use a 'mirrored' region of interest")
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
        assert not args.r
        orderChange = 1
    if args.r:
        assert not args.ho
    if args.b is not None:
        assert approach == 'hessianBased'
    if bool(args.mirror):
        assert mode in ('shallow-water', 'rossby-wave')

    # Choose mode and set parameter values
    op = Options(mode=mode,
                 approach=approach,
                 gradate=True if approach in ('DWP', 'DWR') and mode == 'tohoku' else False,
                 # gradate=False,
                 # gradate=True,
                 plotpvd=True if args.o else False,
                 orderChange=orderChange,
                 refinedSpace=True if args.r else False,
                 bAdapt=bool(args.b) if args.b is not None else False)

    # Establish filenames
    filename = 'outdata/' + mode + '/' + approach
    if args.ho:
        op.orderChange = 1
        filename += '_ho'
    elif args.r:
        op.refinedSpace = True
        filename += '_r'
    if args.b:
        filename += '_b'
    filename += '_' + date
    errorFile = open(filename + '.txt', 'w+')
    files = {}
    extensions = ['Integrand']
    if op.mode == 'tohoku':
        extensions.append('P02')
        extensions.append('P06')
    else:
        extensions.append('Integrand-mirrored')
    for e in extensions:
        files[e] = open(filename + e + '.txt', 'w+')

    # Get data and save to disk
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    Jlist = np.zeros(len(resolutions))
    for i in resolutions:
        quantities = solver(i, regen=bool(args.regen), mirror=bool(args.mirror), op=op)
        rel = np.abs(op.J - quantities['J_h']) / np.abs(op.J)
        if op.mode == "rossby-wave":
            quantities["Mirrored OF error"] = np.abs(op.J_mirror - quantities['J_h mirrored']) / np.abs(op.J_mirror)
        print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs OF error: %.4e"
              % (i, quantities['meanElements'], quantities['J_h'], quantities['solverTimer'], rel))
        errorFile.write('%d, %.4e' % (quantities['meanElements'], rel))
        for tag in ("peak", "dist", "spd", "TV P02", "TV P06", "J_h mirrored", "Mirrored OF error"):
            if tag in quantities:
                errorFile.write(", %.4e" % quantities[tag])
        errorFile.write(", %.1f, %.4e\n" % (quantities['solverTimer'], quantities['J_h']))
        for tag in files:
            files[tag].writelines(["%s," % val for val in quantities[tag]])
            files[tag].write("\n")
        if approach in ("DWP", "DWR"):
            print("Time for final run: %.1fs" % quantities['adaptSolveTimer'])
    for tag in files:
        files[tag].close()
    errorFile.close()
