from thetis import *
from firedrake.petsc import PETSc

from time import clock

from utils.adaptivity import *
from utils.callbacks import AdvectionCallback
from utils.error_estimators import difference_quotient_estimator, local_norm
from utils.interpolation import interp, mixed_pair_interp
from utils.misc import extract_slice, index_string
from utils.setup import problem_domain


__all__ = ["advect"]


def FixedMesh(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')

    # Set up solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.simulation_export_time = op.timestep * op.timesteps_per_export
    options.simulation_end_time = op.simulation_end_time - 0.5 * op.timestep
    options.timestepper_type = op.timestepper
    options.timestepper_options.solver_parameters_tracer = op.solver_parameters
    PETSc.Sys.Print("Using solver parameters %s" % options.timestepper_options.solver_parameters_tracer)
    options.timestep = op.timestep
    options.output_directory = op.directory()
    if not op.plot_pvd:
        options.no_exports = True
    else:
        options.fields_to_export = ['tracer_2d']
    options.horizontal_velocity_scale = op.u_mag
    options.fields_to_export_hdf5 = ['tracer_2d']
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = diffusivity
    options.use_lax_friedrichs_tracer = False
    options.tracer_family = op.tracer_family
    if op.tracer_family == 'cg':
        options.use_limiter_for_tracers = False
    options.use_supg_tracer = op.supg              # NOTE: In development
    options.tracer_source_2d = source
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = AdvectionCallback(solver_obj, parameters=op)
    solver_obj.add_callback(cb1, 'timestep')
    cb2 = callback.DetectorsCallback(solver_obj,
                                     op.h_slice,
                                     ['tracer_2d'],
                                     'horizontal slice',
                                     ["h_slice{i:d}".format(i=i) for i in range(len(op.h_slice))],
                                     export_to_hdf5=True)
    solver_obj.add_callback(cb2, 'export')
    # cb3 = callback.DetectorsCallback(solver_obj,
    #                                  op.v_slice,
    #                                  ['tracer_2d'],
    #                                  'vertical slice',
    #                                  ["v_slice{i:d}".format(i=i) for i in range(len(op.v_slice))],
    #                                  export_to_hdf5=True)
    # solver_obj.add_callback(cb3, 'export')
    solver_obj.bnd_functions = BCs

    # Solve and extract timeseries / functionals
    quantities = {}
    solver_timer = clock()
    solver_obj.iterate()
    solver_timer = clock() - solver_timer
    quantities['J_h'] = cb1.get_val()          # Evaluate objective functional

    extract_slice(quantities, direction='h', op=op)
    # extract_slice(quantities, direction='v', op=op)

    # Output mesh statistics and solver times
    quantities['mean_elements'] = mesh.num_cells()
    quantities['solver_timer'] = solver_timer
    quantities['adapt_solve_timer'] = 0.

    return quantities


def HessianBased(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    if op.plot_metric:
        metric_file = File(op.directory() + "Metric2d.pvd")

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

    adapt_solve_timer = 0.
    quantities = {}
    while cnt < op.final_index():
        adapt_timer = clock()
        P1 = FunctionSpace(mesh, "CG", 1)

        tracer = Function(P1).interpolate(tracer_2d)
        for l in range(op.num_adapt): # TODO: Test this functionality

            # Construct metric
            if cnt != 0:   # Can't adapt to zero concentration
                M = steady_metric(tracer, op=op)

            # Adapt mesh and interpolate variables
            if cnt != 0:
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            if l < op.num_adapt-1:
                tracer = interp(mesh, tracer)

        if (cnt != 0) and (op.num_adapt != 0):
            if op.plot_metric:
                M.rename('metric_2d')
                metric_file.write(M, time=t)

            elev_2d, uv_2d, tracer_2d = interp(mesh, elev_2d, uv_2d, tracer_2d)
            b, BCs, source, diffusivity = problem_domain(mesh=mesh, op=op)[3:]     # TODO: find a different way to reset these
        adapt_timer = clock() - adapt_timer

        # Solver object and equations
        adaptive_solver_obj = solver2d.FlowSolver2d(mesh, b)
        adaptive_options = adaptive_solver_obj.options
        adaptive_options.element_family = op.family
        adaptive_options.use_nonlinear_equations = True
        adaptive_options.simulation_export_time = op.timestep * op.timesteps_per_export
        adaptive_options.simulation_end_time = t + op.timestep * (op.timesteps_per_remesh - 0.5)
        adaptive_options.timestepper_type = op.timestepper
        adaptive_options.timestepper_options.solver_parameters_tracer = op.solver_parameters
        PETSc.Sys.Print("Using solver parameters %s" % adaptive_options.timestepper_options.solver_parameters_tracer)
        adaptive_options.timestep = op.timestep
        adaptive_options.output_directory = op.directory()
        if not op.plot_pvd:
            adaptive_options.no_exports = True
        else:
            adaptive_options.fields_to_export = ['tracer_2d']
        adaptive_options.horizontal_velocity_scale = op.u_mag
        adaptive_options.fields_to_export_hdf5 = ['tracer_2d']
        adaptive_options.solve_tracer = True
        adaptive_options.tracer_only = True  # Need use tracer-only branch to use this functionality
        adaptive_options.horizontal_diffusivity = diffusivity
        adaptive_options.use_lax_friedrichs_tracer = False
        adaptive_options.tracer_family = op.tracer_family
        if op.tracer_family == 'cg':
            adaptive_options.use_limiter_for_tracers = False
        adaptive_options.use_supg_tracer = op.supg              # NOTE: In development
        adaptive_options.tracer_source_2d = source
        adaptive_solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d, tracer=tracer_2d)
        adaptive_solver_obj.i_export = int(cnt / op.timesteps_per_export)
        adaptive_solver_obj.next_export_t = adaptive_solver_obj.i_export * adaptive_options.simulation_export_time
        adaptive_solver_obj.iteration = cnt
        adaptive_solver_obj.simulation_time = t
        for e in adaptive_solver_obj.exporters.values():
            e.set_next_export_ix(adaptive_solver_obj.i_export)

        # Establish callbacks and iterate
        cb1 = AdvectionCallback(adaptive_solver_obj, parameters=op)
        if cnt != 0:
            cb1.integrant = quantities['J_h']
            cb1.old_value = old_val
        adaptive_solver_obj.add_callback(cb1, 'timestep')
        cb2 = callback.DetectorsCallback(adaptive_solver_obj,
                                         op.h_slice,
                                         ['tracer_2d'],
                                         'horizontal slice',
                                         ["h_slice{i:d}".format(i=i) for i in range(len(op.h_slice))],
                                         export_to_hdf5=True)
        adaptive_solver_obj.add_callback(cb2, 'export')
        # cb3 = callback.DetectorsCallback(adaptive_solver_obj,
        #                                  op.v_slice,
        #                                  ['tracer_2d'],
        #                                  'vertical slice',
        #                                  ["v_slice{i:d}".format(i=i) for i in range(len(op.v_slice))],
        #                                  export_to_hdf5=True)
        # adaptive_solver_obj.add_callback(cb3, 'export')
        adaptive_solver_obj.bnd_functions = BCs
        solver_timer = clock()
        adaptive_solver_obj.iterate()
        solver_timer = clock() - solver_timer
        quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
        old_val = cb1.old_value
        extract_slice(quantities, direction='h', op=op)
        # extract_slice(quantities, direction='v', op=op)

        # Get mesh stats
        nEle = mesh.num_cells()
        mM = [min(nEle, mM[0]), max(nEle, mM[1])]
        Sn += nEle
        cnt += op.timesteps_per_remesh
        t += op.timestep * op.timesteps_per_remesh
        av = op.adaptation_stats(int(cnt/op.timesteps_per_remesh+1), adapt_timer, solver_timer, nEle, Sn, mM, cnt * op.timestep)
        adapt_solve_timer += adapt_timer + solver_timer

        # Extract fields for next step
        uv_2d, elev_2d = adaptive_solver_obj.fields.solution_2d.split()
        tracer_2d = adaptive_solver_obj.fields.tracer_2d

    # Output mesh statistics and solver times
    quantities['mean_elements'] = av
    quantities['solver_timer'] = adapt_solve_timer
    quantities['adapt_solve_timer'] = adapt_solve_timer

    return quantities


from thetis_adjoint import *
import pyadjoint
from fenics_adjoint.solving import SolveBlock                                       # For extracting adjoint solutions


def DWP(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    if op.plot_metric:
        metric_file = File(op.directory() + "Metric2d.pvd")

    init_timer = clock()
    if op.plot_pvd:
        error_file = File(op.directory() + "ErrorIndicator2d.pvd")
        adjoint_file = File(op.directory() + "Adjoint2d.pvd")

    # Initialise domain and physical parameters
    P1 = FunctionSpace(mesh, "CG", 1)
    P1DG = FunctionSpace(mesh, "DG", 1)
    tracer_space = P1DG if op.tracer_family == 'dg' else P1
    tracer_2d = Function(tracer_space, name='tracer_2d')

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(tracer_space, name='adjoint_2d')
    epsilon = Function(P1, name='error_2d')
    epsilon_ = Function(P1)

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle

    # # Get initial boundary metric
    # if op.gradate:
    #     H0 = Function(P1).interpolate(CellSize(mesh))

    if not regen:

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.simulation_export_time = op.timestep * op.timesteps_per_remesh
        options.simulation_end_time = op.simulation_end_time - 0.5 * op.timestep
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters_tracer = op.solver_parameters
        PETSc.Sys.Print("Using solver parameters %s" % options.timestepper_options.solver_parameters_tracer)
        options.timestep = op.timestep
        options.output_directory = op.directory()
        options.fields_to_export = ['tracer_2d']
        options.fields_to_export_hdf5 = ['tracer_2d']
        options.solve_tracer = True
        options.tracer_only = True
        options.horizontal_diffusivity = diffusivity
        options.use_lax_friedrichs_tracer = False
        options.tracer_family = op.tracer_family
        if op.tracer_family == 'cg':
            options.use_limiter_for_tracers = False
        options.use_supg_tracer = op.supg              # NOTE: In development
        options.tracer_source_2d = source
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = AdvectionCallback(solver_obj, parameters=op)
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions = BCs
        init_timer = clock() - init_timer
        PETSc.Sys.Print('Problem initialised. Setup time: %.3fs' % init_timer)

        primal_timer = clock()
        solver_obj.iterate()
        primal_timer = clock() - primal_timer
        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        PETSc.Sys.Print('Primal run complete. Solver time: %.3fs' % primal_timer)

        # Compute gradient
        gradient_timer = clock()
        compute_gradient(J, Control(diffusivity))   # TODO: Gradient w.r.t. some fields is more costly than others...
        gradient_timer = clock() - gradient_timer

        # Extract adjoint solutions
        dual_timer = clock()
        tape = get_working_tape()
        # tape.visualise(open_in_browser=True)
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock) and block.adj_sol is not None]
        N = len(solve_blocks)
        r = N % op.timesteps_per_export                            # Number of extra tape annotations in setup
        for i in range(N - 1, r - 1, -op.timesteps_per_export):
            dual.assign(solve_blocks[i].adj_sol)
            index_str = index_string(int((i - r) / op.timesteps_per_export))
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str, mode=FILE_CREATE) as sa:
                sa.store(dual)
                sa.close()
            if op.plot_pvd:
                adjoint_file.write(dual, time=op.timestep * (i - r))
        dual_timer = clock() - dual_timer
        PETSc.Sys.Print('Dual run complete. Run time: %.3fs' % dual_timer)

    tape.clear_tape()
    with pyadjoint.stop_annotating():

        error_timer = clock()
        for k in range(0, op.final_mesh_index()):  # Loop back over times to generate error estimators
            PETSc.Sys.Print('Generating error estimate %d / %d' % (k + 1, op.final_mesh_index()))
            with DumbCheckpoint(op.directory() + 'hdf5/Tracer2d_' + index_string(k), mode=FILE_READ) as lv:
                lv.load(tracer_2d)
                lv.close()

            # Load adjoint data and form indicators
            epsilon.interpolate(tracer_2d * dual)
            for i in range(k, min(k + op.final_export() - op.first_export(), op.final_export())):
                with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_string(i), mode=FILE_READ) as la:
                    la.load(dual)
                    la.close()
                epsilon_.interpolate(tracer_2d * dual)
                epsilon = pointwise_max(epsilon, epsilon_)
            epsilon = normalise_indicator(epsilon, op=op)
            epsilon.rename('error_2d')
            with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_string(k), mode=FILE_CREATE) as se:
                se.store(epsilon)
                se.close()
            if op.plot_pvd:
                error_file.write(epsilon, time=float(k))
        error_timer = clock() - error_timer
        PETSc.Sys.Print('Errors estimated. Run time: %.3fs' % error_timer)

        # Run adaptive primal run
        cnt = 0
        adapt_solve_timer = 0.
        t = 0.
        tracer_2d.assign(0.)
        quantities = {}
        # bdy = 'on_boundary'
        uv_2d, elev_2d = Function(op.mixed_space(mesh)).split()
        uv_2d.interpolate(u0)
        elev_2d.interpolate(eta0)
        while cnt < op.final_index():
            adapt_timer = clock()
            if cnt != 0:    # Do not adapt to initial zero concentration
                for l in range(op.num_adapt):  # TODO: Test this functionality

                    # Construct metric
                    index_str = index_string(int(cnt / op.timesteps_per_remesh))
                    with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_str, mode=FILE_READ) as le:
                        le.load(epsilon)
                        le.close()
                    estimate = Function(FunctionSpace(mesh, "CG", 1)).assign(interp(mesh, epsilon))
                    M = isotropic_metric(estimate, invert=False, op=op)
                    if op.gradate:
                        # M_ = isotropic_metric(interp(mesh, H0), bdy=bdy, op=op)  # Initial boundary metric
                        # M = metric_intersection(M, M_, bdy=bdy)
                        gradate_metric(M, op=op)

                    # Adapt mesh and interpolate variables
                    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh

                if op.num_adapt != 0:
                    if op.plot_metric:
                        M.rename('metric_2d')
                        metric_file.write(M, time=t)
                    tracer_2d = interp(mesh, tracer_2d)
                    u0, eta0, b, BCs, source, diffusivity = problem_domain(mesh=mesh, op=op)[1:] # TODO: find a different way to reset these
                    uv_2d, elev_2d = Function(op.mixed_space(mesh)).split()
                    elev_2d.interpolate(eta0)
                    uv_2d.interpolate(u0)
            adapt_timer = clock() - adapt_timer

            # Solver object and equations
            adaptive_solver_obj = solver2d.FlowSolver2d(mesh, b)
            adaptive_options = adaptive_solver_obj.options
            adaptive_options.element_family = op.family
            adaptive_options.use_nonlinear_equations = True
            adaptive_options.simulation_export_time = op.timestep * op.timesteps_per_export
            adaptive_options.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adaptive_options.timestepper_type = op.timestepper
            adaptive_options.timestepper_options.solver_parameters_tracer = op.solver_parameters
            PETSc.Sys.Print("Using solver parameters %s" % adaptive_options.timestepper_options.solver_parameters)
            adaptive_options.timestep = op.timestep
            adaptive_options.output_directory = op.directory()
            if not op.plot_pvd:
                adaptive_options.no_exports = True
            else:
                adaptive_options.fields_to_export = ['tracer_2d']
            adaptive_options.horizontal_velocity_scale = op.u_mag
            adaptive_options.fields_to_export = ['tracer_2d']
            adaptive_options.fields_to_export_hdf5 = ['tracer_2d']
            adaptive_options.solve_tracer = True
            adaptive_options.tracer_only = True  # Need use tracer-only branch to use this functionality
            adaptive_options.horizontal_diffusivity = diffusivity
            adaptive_options.use_lax_friedrichs_tracer = False
            adaptive_options.tracer_family = op.tracer_family
            if op.tracer_family == 'cg':
                adaptive_options.use_limiter_for_tracers = False
            adaptive_options.use_supg_tracer = op.supg              # NOTE: In development
            adaptive_options.tracer_source_2d = source
            adaptive_solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d, tracer=tracer_2d)
            adaptive_solver_obj.i_export = int(cnt / op.timesteps_per_export)
            adaptive_solver_obj.next_export_t = adaptive_solver_obj.i_export * adaptive_options.simulation_export_time
            adaptive_solver_obj.iteration = cnt
            adaptive_solver_obj.simulation_time = t
            for e in adaptive_solver_obj.exporters.values():
                e.set_next_export_ix(adaptive_solver_obj.i_export)

            # Evaluate callbacks and iterate
            cb1 = AdvectionCallback(adaptive_solver_obj, parameters=op)
            if cnt != 0:
                cb1.integrant = quantities['J_h']
                cb1.old_value = old_val
            adaptive_solver_obj.add_callback(cb1, 'timestep')
            cb2 = callback.DetectorsCallback(adaptive_solver_obj,
                                             op.h_slice,
                                             ['tracer_2d'],
                                             'horizontal slice',
                                             ["h_slice{i:d}".format(i=i) for i in range(len(op.h_slice))],
                                             export_to_hdf5=True)
            adaptive_solver_obj.add_callback(cb2, 'export')
            # cb3 = callback.DetectorsCallback(adaptive_solver_obj,
            #                                  op.v_slice,
            #                                  ['tracer_2d'],
            #                                  'vertical slice',
            #                                  ["v_slice{i:d}".format(i=i) for i in range(len(op.v_slice))],
            #                                  export_to_hdf5=True)
            # adaptive_solver_obj.add_callback(cb3, 'export')
            adaptive_solver_obj.bnd_functions = BCs
            solver_timer = clock()
            adaptive_solver_obj.iterate()
            solver_timer = clock() - solver_timer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
            old_val = cb1.old_value
            extract_slice(quantities, direction='h', op=op)
            # extract_slice(quantities, direction='v', op=op)

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adapt_timer, solver_timer, nEle, Sn, mM, cnt * op.timestep)
            adapt_solve_timer += adapt_timer + solver_timer

            # Extract fields for next solver block
            tracer_2d = adaptive_solver_obj.fields.tracer_2d

        # Output mesh statistics and solver times
        total_timer = error_timer + adapt_solve_timer
        if not regen:
            total_timer += primal_timer + gradient_timer + dual_timer
        quantities['mean_elements'] = av
        quantities['solver_timer'] = total_timer
        quantities['adapt_solve_timer'] = adapt_solve_timer

        return quantities


def DWR(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    if op.plot_metric:
        metric_file = File(op.directory() + "Metric2d.pvd")

    init_timer = clock()
    if op.plot_pvd:
        residual_file = File(op.directory() + "Residual2d.pvd")
        error_file = File(op.directory() + "ErrorIndicator2d.pvd")
        adjoint_file = File(op.directory() + "Adjoint2d.pvd")

    # Initialise domain and physical parameters
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)
    P1DG = FunctionSpace(mesh, "DG", 1)
    tracer_space = P1DG if op.tracer_family == 'dg' else P1
    tracer_2d = Function(tracer_space)

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(tracer_space, name='adjoint_2d')
    epsilon = Function(P1, name='error_2d')

    if op.order_increase:
        duale = Function(FunctionSpace(mesh, "DG" if op.tracer_family == 'dg' else "CG", 2))
        residual_2d = Function(tracer_space)
    else:
        dual_old = Function(tracer_space, name='adjoint_old')
        residual_2d = Function(P0)

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle

    # # Get initial boundary metric
    # if op.gradate:
    #     H0 = Function(P1).interpolate(CellSize(mesh))

    if not regen:

        # Solve fixed mesh primal problem to get residuals and adjoint solutions
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True
        options.simulation_export_time = op.timestep * op.timesteps_per_export
        options.simulation_end_time = op.simulation_end_time - 0.5 * op.timestep
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters_tracer = op.solver_parameters
        PETSc.Sys.Print("Using solver parameters %s" % options.timestepper_options.solver_parameters)
        options.timestep = op.timestep
        options.output_directory = op.directory()   # Need this for residual callback
        options.export_diagnostics = False
        options.solve_tracer = True
        options.tracer_only = True
        options.horizontal_diffusivity = diffusivity
        options.use_lax_friedrichs_tracer = False
        options.tracer_family = op.tracer_family
        if op.tracer_family == 'cg':
            options.use_limiter_for_tracers = False
        options.use_supg_tracer = op.supg              # NOTE: In development
        options.tracer_source_2d = source
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = AdvectionCallback(solver_obj, parameters=op)
        if op.order_increase:
            cb2 = callback.CellResidualCallback(solver_obj, export_to_hdf5=True)
        else:
            cb2 = callback.ExplicitErrorCallback(solver_obj, export_to_hdf5=True)
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.add_callback(cb2, 'export')
        solver_obj.bnd_functions = BCs
        init_timer = clock() - init_timer
        PETSc.Sys.Print('Problem initialised. Setup time: %.3fs' % init_timer)

        primal_timer = clock()
        solver_obj.iterate()
        primal_timer = clock() - primal_timer
        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        PETSc.Sys.Print('Primal run complete. Solver time: %.3fs' % primal_timer)

        # Compute gradient
        gradient_timer = clock()
        compute_gradient(J, Control(diffusivity))   # TODO: Gradient w.r.t. some fields is more costly than others...
        gradient_timer = clock() - gradient_timer

        # Extract adjoint solutions
        dual_timer = clock()
        tape = get_working_tape()
        # tape.visualise(open_in_browser=True)
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock) and block.adj_sol is not None]
        N = len(solve_blocks)
        r = N % op.timesteps_per_remesh                       # Number of extra tape annotations in setup
        for i in range(r, N, op.timesteps_per_remesh):        # Iterate r is the first timestep
            dual.assign(solve_blocks[i].adj_sol)
            index_str = index_string(int((i - r) / op.timesteps_per_remesh))
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str,  mode=FILE_CREATE) as sa:
                sa.store(dual)
                sa.close()
            if not op.order_increase:
                if i == r:
                    dual_old.assign(solve_blocks[i].adj_sol)
                else:
                    dual_old.assign(solve_blocks[i-1].adj_sol)
                with DumbCheckpoint(op.directory() + 'hdf5/PreviousAdjoint2d_' + index_str, mode=FILE_CREATE) as so:
                    so.store(dual_old)
                    so.close()
            if op.plot_pvd:
                adjoint_file.write(dual, time=op.timestep * (i - r))
        dual_timer = clock() - dual_timer
        PETSc.Sys.Print('Dual run complete. Run time: %.3fs' % dual_timer)

        tape.clear_tape()
        with pyadjoint.stop_annotating():

            residuals = []
            error_timer = clock()
            for k in range(0, int(op.final_index() / op.timesteps_per_export)):
                PETSc.Sys.Print('Generating error estimate %d / %d'
                      % (int(k/op.exports_per_remesh()) + 1, int(op.final_index() / op.timesteps_per_remesh)))

                # Load residuals
                tag = 'CellResidual2d_' if op.order_increase else 'ExplicitError2d_'
                with DumbCheckpoint(op.directory() + 'hdf5/' + tag + index_string(k), mode=FILE_READ) as lr:
                    if op.order_increase:
                        lr.load(residual_2d, name="residual")
                    else:
                        lr.load(residual_2d, name="explicit error")
                    lr.close()

                residuals.append(residual_2d)   # TODO: This is grossly inefficient. Just load from HDF5
                if k % op.exports_per_remesh() == op.exports_per_remesh()-1:

                    # L-inf
                    for i in range(1, len(residuals)):
                        residual_2d = pointwise_max(residual_2d, residuals[i])

                    # # L1
                    # residual_2d.interpolate(op.timestep * sum(abs(residuals[i] + residuals[i-1]) for i in range(1, op.exports_per_remesh())))

                    # # Time integrate residual over current 'window'
                    # residual_2d.interpolate(op.timestep * sum(residuals[i] + residuals[i-1] for i in range(1, op.exports_per_remesh())))

                    residuals = []
                    if op.plot_pvd:
                        residual_file.write(residual_2d, time=float(op.timestep * op.timesteps_per_remesh * (k+1)))

                    # Load adjoint data and form indicators
                    index_str = index_string(int((k+1)/op.exports_per_remesh()-1))
                    with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str, mode=FILE_READ) as la:
                        la.load(dual)
                        la.close()
                    if op.order_increase:   # TODO: Requires patchwise interpolation to do properly
                        duale.interpolate(dual)
                        epsilon.interpolate(residual_2d * duale)
                    else:
                        with DumbCheckpoint(op.directory() + 'hdf5/PreviousAdjoint2d_' + index_str, mode=FILE_READ) as lo:
                            lo.load(dual_old)
                            lo.close()
                        epsilon.interpolate(difference_quotient_estimator(solver_obj, residual_2d, dual, dual_old))
                    epsilon = normalise_indicator(epsilon, op=op)
                    epsilon.rename('error_2d')
                    with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_str, mode=FILE_CREATE) as se:
                        se.store(epsilon)
                        se.close()
                    if op.plot_pvd:
                        error_file.write(epsilon, time=float(op.timestep * op.timesteps_per_remesh * k))
            error_timer = clock() - error_timer
            PETSc.Sys.Print('Errors estimated. Run time: %.3fs' % error_timer)

    with pyadjoint.stop_annotating():

        # Run adaptive primal run
        cnt = 0
        adapt_solve_timer = 0.
        t = 0.
        quantities = {}
        # bdy = 'on_boundary'
        uv_2d, elev_2d = Function(op.mixed_space(mesh)).split()
        uv_2d.interpolate(u0)
        elev_2d.interpolate(eta0)
        while cnt < op.final_index():
            adapt_timer = clock()
            if cnt != 0:    # Don't adapt to initial zero concentration
                for l in range(op.num_adapt): # TODO: Test this functionality

                    # Construct metric
                    index_str = index_string(int(cnt / op.timesteps_per_remesh))
                    with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_str, mode=FILE_READ) as le:
                        le.load(epsilon)
                        le.close()
                    estimate = Function(FunctionSpace(mesh, "CG", 1)).assign(interp(mesh, epsilon))
                    M = isotropic_metric(estimate, invert=False, op=op)
                    if op.gradate:
                        # M_ = isotropic_metric(interp(mesh, H0), bdy=bdy, op=op)   # Initial boundary metric
                        # M = metric_intersection(M, M_, bdy=bdy)
                        M = gradate_metric(M, op=op)

                    # Adapt mesh and interpolate variables
                    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh

                if op.num_adapt != 0:
                    if op.plot_metric:
                        M.rename('metric_2d')
                        metric_file.write(M, time=t)
                    tracer_2d = interp(mesh, tracer_2d)
                    u0, eta0, b, BCs, source, diffusivity = problem_domain(mesh=mesh, op=op)[1:]  # TODO: find a different way to reset these
                    V = op.mixed_space(mesh)
                    uv_2d, elev_2d = Function(V).split()
                    elev_2d.interpolate(eta0)
                    uv_2d.interpolate(u0)
            adapt_timer = clock() - adapt_timer

            # Solver object and equations
            adaptive_solver_obj = solver2d.FlowSolver2d(mesh, b)
            adaptive_options = adaptive_solver_obj.options
            adaptive_options.element_family = op.family
            adaptive_options.use_nonlinear_equations = True
            adaptive_options.simulation_export_time = op.timestep * op.timesteps_per_export
            adaptive_options.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adaptive_options.timestepper_type = op.timestepper
            adaptive_options.timestepper_options.solver_parameters_tracer = op.solver_parameters
            PETSc.Sys.Print("Using solver parameters %s" % adaptive_options.timestepper_options.solver_parameters)
            adaptive_options.timestep = op.timestep
            adaptive_options.output_directory = op.directory()
            if not op.plot_pvd:
                adaptive_options.no_exports = True
            else:
                adaptive_options.fields_to_export = ['tracer_2d']
            adaptive_options.horizontal_velocity_scale = op.u_mag
            adaptive_options.solve_tracer = True
            adaptive_options.tracer_only = True  # Need use tracer-only branch to use this functionality
            adaptive_options.horizontal_diffusivity = diffusivity
            adaptive_options.use_lax_friedrichs_tracer = False 
            adaptive_options.tracer_family = op.tracer_family
            if op.tracer_family == 'cg':
                adaptive_options.use_limiter_for_tracers = False
            adaptive_options.use_supg_tracer = op.supg              # NOTE: In development
            adaptive_options.tracer_source_2d = source
            adaptive_solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d, tracer=tracer_2d)
            adaptive_solver_obj.i_export = int(cnt / op.timesteps_per_export)
            adaptive_solver_obj.next_export_t = adaptive_solver_obj.i_export * adaptive_options.simulation_export_time
            adaptive_solver_obj.iteration = cnt
            adaptive_solver_obj.simulation_time = t
            for e in adaptive_solver_obj.exporters.values():
                e.set_next_export_ix(adaptive_solver_obj.i_export)

            # Evaluate callbacks and iterate
            cb1 = AdvectionCallback(adaptive_solver_obj, parameters=op)
            if cnt != 0:
                cb1.integrant = quantities['J_h']
                cb1.old_value = old_val
            adaptive_solver_obj.add_callback(cb1, 'timestep')
            cb2 = callback.DetectorsCallback(adaptive_solver_obj,
                                             op.h_slice,
                                             ['tracer_2d'],
                                             'horizontal slice',
                                             ["h_slice{i:d}".format(i=i) for i in range(len(op.h_slice))],
                                             export_to_hdf5=True)
            adaptive_solver_obj.add_callback(cb2, 'export')
            # cb3 = callback.DetectorsCallback(adaptive_solver_obj,
            #                                  op.v_slice,
            #                                  ['tracer_2d'],
            #                                  'vertical slice',
            #                                  ["v_slice{i:d}".format(i=i) for i in range(len(op.v_slice))],
            #                                  export_to_hdf5=True)
            # adaptive_solver_obj.add_callback(cb3, 'export')
            adaptive_solver_obj.bnd_functions['shallow_water'] = BCs
            solver_timer = clock()
            adaptive_solver_obj.iterate()
            solver_timer = clock() - solver_timer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
            old_val = cb1.old_value
            extract_slice(quantities, direction='h', op=op)
            # extract_slice(quantities, direction='v', op=op)

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adapt_timer, solver_timer, nEle, Sn, mM, cnt * op.timestep)
            adapt_solve_timer += adapt_timer + solver_timer

            # Extract fields for next solver block
            tracer_2d = adaptive_solver_obj.fields.tracer_2d


        # Output mesh statistics and solver times
        total_timer = error_timer + adapt_solve_timer
        if not regen:
            total_timer += primal_timer + gradient_timer + dual_timer
        quantities['mean_elements'] = av
        quantities['solver_timer'] = total_timer
        quantities['adapt_solve_timer'] = adapt_solve_timer

        return quantities


def advect(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    solvers = {'FixedMesh': FixedMesh, 'HessianBased': HessianBased, 'DWP': DWP, 'DWR': DWR}

    return solvers[op.approach](mesh, u0, eta0, b, BCs, source, diffusivity, regen=regen, op=op)
