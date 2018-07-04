from thetis import *

import numpy as np
from time import clock

from utils.adaptivity import *
from utils.callbacks import AdvectionCallback, ObjectiveAdvectionCallback
from utils.interpolation import interp, mixedPairInterp
from utils.setup import problemDomain


__all__ = ["advect"]


def fixedMesh(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')

    # Initialise solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.simulation_export_time = op.timestep * op.timesteps_per_export
    options.simulation_end_time = op.end_time - 0.5 * op.timestep
    options.timestepper_type = op.timestepper
    options.timestepper_options.solver_parameters_tracer = op.solver_parameters
    print("Using solver parameters %s" % options.timestepper_options.solver_parameters_tracer)
    options.timestep = op.timestep
    options.output_directory = op.directory()
    if not op.plot_pvd:
        options.no_exports = True
    options.horizontal_velocity_scale = op.u_mag
    options.fields_to_export = ['tracer_2d']
    options.fields_to_export_hdf5 = ['tracer_2d']
    options.solve_tracer = True
    options.tracer_only = True  # Need use tracer-only branch to use this functionality
    options.horizontal_diffusivity = diffusivity
    options.use_lax_friedrichs_tracer = False                   # TODO: This is a temporary fix
    options.tracer_source_2d = source
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = AdvectionCallback(solver_obj)
    cb1.op = op
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions = BCs

    # Solve and extract timeseries / functionals
    quantities = {}
    solverTimer = clock()
    solver_obj.iterate()
    solverTimer = clock() - solverTimer
    quantities['J_h'] = cb1.get_val()          # Evaluate objective functional

    # Output mesh statistics and solver times
    quantities['meanElements'] = mesh.num_cells()
    quantities['solverTimer'] = solverTimer
    quantities['adaptSolveTimer'] = 0.

    return quantities


def hessianBased(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    if op.plot_metric:
        mFile = File(op.directory() + "Metric2d.pvd")

    # Initialise domain and physical parameters
    V = op.mixed_space(mesh)
    uv_2d, elev_2d = Function(V).split()  # Needed to load data into
    elev_2d.interpolate(eta0)
    uv_2d.interpolate(u0)
    tracer_2d = Function(FunctionSpace(mesh, "CG", 1))

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling   # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    cnt = 0
    t = 0.

    adaptSolveTimer = 0.
    quantities = {}
    while cnt < op.final_index():
        adaptTimer = clock()
        P1 = FunctionSpace(mesh, "CG", 1)

        tracer = Function(P1).interpolate(tracer_2d)
        for l in range(op.adaptations):                  # TODO: Test this functionality

            # Construct metric
            if cnt != 0:   # Can't adapt to zero concentration
                M = steadyMetric(tracer, op=op)

            # Adapt mesh and interpolate variables
            if cnt != 0:
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            if l < op.adaptations-1:
                tracer = interp(mesh, tracer)

        if cnt != 0:
            if op.plot_metric:
                if op.adaptations == 0:
                    M = steadyMetric(tracer, op=op)
                M.rename('metric_2d')
                mFile.write(M, time=t)

            elev_2d, uv_2d, tracer_2d = interp(mesh, elev_2d, uv_2d, tracer_2d)
            b, BCs, source, diffusivity = problemDomain(mesh=mesh, op=op)[3:]     # TODO: find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
            tracer_2d.rename('tracer_2d')
        adaptTimer = clock() - adaptTimer

        # Solver object and equations
        adapSolver = solver2d.FlowSolver2d(mesh, b)
        adapOpt = adapSolver.options
        adapOpt.element_family = op.family
        adapOpt.use_nonlinear_equations = True
        adapOpt.simulation_export_time = op.timestep * op.timesteps_per_export
        adapOpt.simulation_end_time = t + op.timestep * (op.timesteps_per_remesh - 0.5)
        adapOpt.timestepper_type = op.timestepper
        adapOpt.timestepper_options.solver_parameters_tracer = op.solver_parameters
        print("Using solver parameters %s" % adapOpt.timestepper_options.solver_parameters_tracer)
        adapOpt.timestep = op.timestep
        adapOpt.output_directory = op.directory()
        if not op.plot_pvd:
            adapOpt.no_exports = True
        adapOpt.horizontal_velocity_scale = op.u_mag
        adapOpt.fields_to_export = ['tracer_2d']
        adapOpt.fields_to_export_hdf5 = ['tracer_2d']
        adapOpt.solve_tracer = True
        adapOpt.tracer_only = True  # Need use tracer-only branch to use this functionality
        adapOpt.horizontal_diffusivity = diffusivity
        adapOpt.use_lax_friedrichs_tracer = False                   # TODO: This is a temporary fix
        # adapOpt.use_lax_friedrichs_tracer = True
        adapOpt.tracer_source_2d = source
        adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d, tracer=tracer_2d)
        adapSolver.i_export = int(cnt / op.timesteps_per_export)
        adapSolver.next_export_t = adapSolver.i_export * adapSolver.options.simulation_export_time
        adapSolver.iteration = cnt
        adapSolver.simulation_time = t
        for e in adapSolver.exporters.values():
            e.set_next_export_ix(adapSolver.i_export)

        # Establish callbacks and iterate
        cb1 = AdvectionCallback(adapSolver)
        cb1.op = op
        if cnt != 0:
            cb1.old_value = quantities['J_h']
        adapSolver.add_callback(cb1, 'timestep')
        adapSolver.bnd_functions = BCs
        solverTimer = clock()
        adapSolver.iterate()
        solverTimer = clock() - solverTimer
        quantities['J_h'] = cb1.get_val()  # Evaluate objective functional

        # Get mesh stats
        nEle = mesh.num_cells()
        mM = [min(nEle, mM[0]), max(nEle, mM[1])]
        Sn += nEle
        cnt += op.timesteps_per_remesh
        t += op.timestep * op.timesteps_per_remesh
        av = op.adaptation_stats(int(cnt/op.timesteps_per_remesh+1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.timestep)
        adaptSolveTimer += adaptTimer + solverTimer

        # Extract fields for next step
        uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
        tracer_2d = adapSolver.fields.tracer_2d

    # Output mesh statistics and solver times
    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['adaptSolveTimer'] = adaptSolveTimer

    return quantities


from thetis_adjoint import *
import pyadjoint
from fenics_adjoint.solving import SolveBlock                                       # For extracting adjoint solutions


def DWP(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    if op.plot_metric:
        mFile = File(op.directory() + "Metric2d.pvd")

    initTimer = clock()
    if op.plot_pvd:
        errorFile = File(op.directory() + "ErrorIndicator2d.pvd")
        adjointFile = File(op.directory() + "Adjoint2d.pvd")

    # Initialise domain and physical parameters
    P1 = FunctionSpace(mesh, "CG", 1)
    tracer_2d = Function(P1, name='tracer_2d')

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(P1, name='adjoint_2d')
    epsilon = Function(P1, name='error_2d')
    epsilon_ = Function(P1)

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle

    # Get initial boundary metric
    if op.gradate:
        H0 = Function(P1).interpolate(CellSize(mesh))

    if not regen:

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.simulation_export_time = op.timestep * op.timesteps_per_remesh
        options.simulation_end_time = op.end_time - 0.5 * op.timestep
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters_tracer = op.solver_parameters
        print("Using solver parameters %s" % options.timestepper_options.solver_parameters_tracer)
        options.timestep = op.timestep
        options.output_directory = op.directory()
        options.fields_to_export = ['tracer_2d']
        options.fields_to_export_hdf5 = ['tracer_2d']
        options.solve_tracer = True
        options.tracer_only = True  # Need use tracer-only branch to use this functionality
        options.horizontal_diffusivity = diffusivity
        options.use_lax_friedrichs_tracer = False  # TODO: This is a temporary fix
        options.tracer_source_2d = source
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveAdvectionCallback(solver_obj)
        cb1.op = op
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)
        primalTimer = clock()
        solver_obj.iterate()
        primalTimer = clock() - primalTimer
        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        compute_gradient(J, Control(diffusivity))
        gradientTimer = clock() - gradientTimer

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.timesteps_per_export                            # Number of extra tape annotations in setup
        for i in range(N - 1, r - 1, -op.timesteps_per_export):
            dual.assign(solve_blocks[i].adj_sol)
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + indexString(int((i - r) / op.timesteps_per_export)), mode=FILE_CREATE) as saveAdj:
                saveAdj.store(dual)
                saveAdj.close()
            if op.plot_pvd:
                adjointFile.write(dual, time=op.timestep * (i - r))
        dualTimer = clock() - dualTimer
        print('Dual run complete. Run time: %.3fs' % dualTimer)

    with pyadjoint.stop_annotating():

        errorTimer = clock()
        for k in range(0, op.final_mesh_index()):  # Loop back over times to generate error estimators
            print('Generating error estimate %d / %d' % (k + 1, op.final_mesh_index()))
            with DumbCheckpoint(op.directory() + 'hdf5/Tracer2d_' + indexString(k), mode=FILE_READ) as loadVel:
                loadVel.load(tracer_2d)
                loadVel.close()

            # Load adjoint data and form indicators
            epsilon.interpolate(tracer_2d * dual)
            for i in range(k, min(k + op.final_export() - op.first_export(), op.final_export())):
                with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual)
                    loadAdj.close()
                epsilon_.interpolate(tracer_2d * dual)
                epsilon = pointwiseMax(epsilon, epsilon_)
            epsilon = normaliseIndicator(epsilon, op=op)
            with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plot_pvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        print('Errors estimated. Run time: %.3fs' % errorTimer)

        # Run adaptive primal run
        cnt = 0
        adaptSolveTimer = 0.
        t = 0.
        tracer_2d.assign(0.)
        quantities = {}
        bdy = 'on_boundary'
        while cnt < op.final_index():
            adaptTimer = clock()
            for l in range(op.adaptations):                                  # TODO: Test this functionality

                # Construct metric
                indexStr = indexString(int(cnt / op.timesteps_per_remesh))
                with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + indexStr, mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(interp(mesh, epsilon))
                M = isotropicMetric(errEst, invert=False, op=op)
                if op.gradate:
                    M_ = isotropicMetric(interp(mesh, H0), bdy=bdy, op=op)  # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=bdy)
                    metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh

            if op.adaptations != 0 and op.plot_metric:
                M.rename('metric_2d')
                mFile.write(M, time=t)
            u0, eta0, b, BCs, source, diffusivity = problemDomain(mesh=mesh, op=op)[1:] # TODO: find a different way to reset these
            V = op.mixed_space(mesh)
            uv_2d, elev_2d = Function(V).split()
            elev_2d.interpolate(eta0)
            uv_2d.interpolate(u0)
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            adapOpt.simulation_export_time = op.timestep * op.timesteps_per_export
            adapOpt.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestepper_options.solver_parameters_tracer = op.solver_parameters
            print("Using solver parameters %s" % adapOpt.timestepper_options.solver_parameters)
            adapOpt.timestep = op.timestep
            adapOpt.output_directory = op.directory()
            if not op.plot_pvd:
                adapOpt.no_exports = True
            adapOpt.horizontal_velocity_scale = op.u_mag
            adapOpt.fields_to_export = ['tracer_2d']
            adapOpt.fields_to_export_hdf5 = ['tracer_2d']
            adapOpt.solve_tracer = True
            adapOpt.tracer_only = True  # Need use tracer-only branch to use this functionality
            adapOpt.horizontal_diffusivity = diffusivity
            adapOpt.use_lax_friedrichs_tracer = False  # TODO: This is a temporary fix
            adapOpt.tracer_source_2d = source
            adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d, tracer=tracer_2d)
            adapSolver.i_export = int(cnt / op.timesteps_per_export)
            adapSolver.next_export_t = adapSolver.i_export * adapSolver.options.simulation_export_time
            adapSolver.iteration = cnt
            adapSolver.simulation_time = t
            for e in adapSolver.exporters.values():
                e.set_next_export_ix(adapSolver.i_export)

            # Evaluate callbacks and iterate
            cb1 = AdvectionCallback(adapSolver)
            cb1.op = op
            if cnt != 0:
                cb1.old_value = quantities['J_h']
            adapSolver.add_callback(cb1, 'timestep')
            adapSolver.bnd_functions = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.timestep)
            adaptSolveTimer += adaptTimer + solverTimer

            # Extract fields for next solver block
            tracer_2d = adapSolver.fields.tracer_2d

        # Output mesh statistics and solver times
        totalTimer = errorTimer + adaptSolveTimer
        if not regen:
            totalTimer += primalTimer + gradientTimer + dualTimer
        quantities['meanElements'] = av
        quantities['solverTimer'] = totalTimer
        quantities['adaptSolveTimer'] = adaptSolveTimer

        return quantities


def advect(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    solvers = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased, 'DWP': DWP}

    return solvers[op.approach](mesh, u0, eta0, b, BCs, source, diffusivity, regen=regen, op=op)
