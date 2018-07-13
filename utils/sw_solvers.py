from thetis import *

import numpy as np
from time import clock

from utils.adaptivity import *
from utils.callbacks import SWCallback, ObjectiveSWCallback
from utils.error_estimators import difference_quotient_estimator
from utils.interpolation import interp
from utils.misc import index_string, peak_and_distance, boundary_region, extract_gauge_data
from utils.setup import problem_domain, RossbyWaveSolution
from utils.timeseries import gauge_total_variation


__all__ = ["tsunami"]


def FixedMesh(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    V = op.mixed_space(mesh)                               # TODO: Parallelise this (and below)
    if op.mode == 'RossbyWave':            # Analytic final-time state
        peak_a, distance_a = peak_and_distance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Initialise solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    options.horizontal_viscosity = diffusivity
    options.use_grad_div_viscosity_term = True              # Symmetric viscous stress
    options.use_lax_friedrichs_velocity = False             # TODO: This is a temporary fix
    options.coriolis_frequency = f
    options.simulation_export_time = op.timestep * op.timesteps_per_export
    options.simulation_end_time = op.end_time - 0.5 * op.timestep
    options.timestepper_type = op.timestepper
    options.timestepper_options.solver_parameters = op.solver_parameters
    print("Using solver parameters %s" % options.timestepper_options.solver_parameters)
    options.timestep = op.timestep
    options.output_directory = op.directory()
    if not op.plot_pvd:
        options.no_exports = True
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = SWCallback(solver_obj)
    cb1.op = op
    if op.mode == 'Tohoku':
        cb2 = callback.DetectorsCallback(solver_obj,
                                         [op.gauge_coordinates(g) for g in op.gauges],
                                         ['elev_2d'],
                                         'timeseries',
                                         op.gauges,
                                         export_to_hdf5=True)
        solver_obj.add_callback(cb2, 'timestep')
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs

    # Solve and extract timeseries / functionals
    quantities = {}
    solver_timer = clock()
    solver_obj.iterate()
    solver_timer = clock() - solver_timer
    quantities['J_h'] = cb1.get_val()          # Evaluate objective functional
    if op.mode == 'Tohoku':
        extract_gauge_data(quantities, op=op)

    # Measure error using metrics, as in Huang et al.     # TODO: Parallelise this (and above)
    if op.mode == 'RossbyWave':
        peak, distance = peak_and_distance(solver_obj.fields.solution_2d.split()[1], op=op)
        distance += 48. # Account for periodic domain
        quantities['peak'] = peak/peak_a
        quantities['dist'] = distance/distance_a
        quantities['spd'] = distance /(op.end_time * 0.4)

    # Output mesh statistics and solver times
    quantities['mean_elements'] = mesh.num_cells()
    quantities['solver_timer'] = solver_timer
    quantities['adapt_solve_timer'] = 0.
    if op.mode == 'Tohoku':
        for g in op.gauges:
            quantities["TV "+g] = gauge_total_variation(quantities[g], gauge=g)

    return quantities


def HessianBased(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    if op.plot_metric:
        mFile = File(op.directory() + "Metric2d.pvd")

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    V = op.mixed_space(mesh)
    uv_2d, elev_2d = Function(V).split()  # Needed to load data into
    elev_2d.interpolate(eta0)
    uv_2d.interpolate(u0)
    if op.mode == 'RossbyWave':    # Analytic final-time state
        peak_a, distance_a = peak_and_distance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling   # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    cnt = 0
    t = 0.

    adapt_solve_timer = 0.
    quantities = {}
    if op.mode == 'Tohoku':
        for g in op.gauges:
            quantities[g] = ()
    while cnt < op.final_index():
        adapt_timer = clock()
        P1 = FunctionSpace(mesh, "CG", 1)

        if op.adapt_field != 's':
            height = Function(P1).interpolate(elev_2d)
        if op.adapt_field != 'f':
            spd = Function(P1).interpolate(sqrt(dot(uv_2d, uv_2d)))
        for l in range(op.adaptations):                  # TODO: Test this functionality

            # Construct metric
            if op.adapt_field != 's':
                M = steady_metric(height, op=op)
            if op.adapt_field != 'f' and cnt != 0:   # Can't adapt to zero velocity
                M2 = steady_metric(spd, op=op)
                if op.adapt_field != 'b':
                    M = M2
                else:
                    try:
                        M = metric_intersection(M, M2)
                    except:
                        print("WARNING: null fluid speed metric")
                        M = metric_intersection(M2, M)
            if op.adapt_on_bathymetry and not (op.adapt_field != 'f' and cnt == 0):
                M2 = steady_metric(b, op=op)
                M = M2 if op.adapt_field != 'f' and cnt == 0. else metric_intersection(M, M2)     # TODO: Convex combination?

            # Adapt mesh and interpolate variables
            if op.adapt_on_bathymetry or cnt != 0 or op.adapt_field == 'f':
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            if l < op.adaptations-1:
                if op.adapt_field != 's':
                    height = interp(mesh, height)
                if op.adapt_field != 'f':
                    spd = interp(mesh, spd)

        if cnt != 0 or op.adapt_field == 'f':
            if op.adaptations != 0 and op.plot_metric:
                M.rename('metric_2d')
                mFile.write(M, time=t)

            elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
            b, BCs, f, diffusivity = problem_domain(mesh=mesh, op=op)[3:]   # TODO: find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
        adapt_timer = clock() - adapt_timer

        # Solver object and equations
        adaptive_solver_obj = solver2d.FlowSolver2d(mesh, b)
        adaptive_options = adaptive_solver_obj.options
        adaptive_options.element_family = op.family
        adaptive_options.use_nonlinear_equations = True
        if diffusivity is not None:
            adaptive_options.horizontal_viscosity = interp(mesh, diffusivity)
        adaptive_options.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
        adaptive_options.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
        adaptive_options.simulation_export_time = op.timestep * op.timesteps_per_export
        adaptive_options.simulation_end_time = t + op.timestep * (op.timesteps_per_remesh - 0.5)
        adaptive_options.timestepper_type = op.timestepper
        adaptive_options.timestepper_options.solver_parameters = op.solver_parameters
        print("Using solver parameters %s" % adaptive_options.timestepper_options.solver_parameters)
        adaptive_options.timestep = op.timestep
        adaptive_options.output_directory = op.directory()
        if not op.plot_pvd:
            adaptive_options.no_exports = True
        adaptive_options.coriolis_frequency = f
        adaptive_solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
        adaptive_solver_obj.i_export = int(cnt / op.timesteps_per_export)
        adaptive_solver_obj.next_export_t = adaptive_solver_obj.i_export * adaptive_options.simulation_export_time
        adaptive_solver_obj.iteration = cnt
        adaptive_solver_obj.simulation_time = t
        for e in adaptive_solver_obj.exporters.values():
            e.set_next_export_ix(adaptive_solver_obj.i_export)

        # Establish callbacks and iterate
        cb1 = SWCallback(adaptive_solver_obj)
        cb1.op = op
        if cnt != 0:
            cb1.old_value = quantities['J_h']
        adaptive_solver_obj.add_callback(cb1, 'timestep')
        if op.mode == 'Tohoku':
            cb2 = callback.DetectorsCallback(adaptive_solver_obj,
                                             [op.gauge_coordinates(g) for g in op.gauges],
                                             ['elev_2d'],
                                             'timeseries',
                                             op.gauges,
                                             export_to_hdf5=True)
            adaptive_solver_obj.add_callback(cb2, 'timestep')
        adaptive_solver_obj.bnd_functions['shallow_water'] = BCs
        solver_timer = clock()
        adaptive_solver_obj.iterate()
        solver_timer = clock() - solver_timer
        quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
        if op.mode == 'Tohoku':
            extract_gauge_data(quantities, op=op)

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

    # Measure error using metrics, as in Huang et al.
    if op.mode == 'RossbyWave':
        peak, distance = peak_and_distance(elev_2d, op=op)
        quantities['peak'] = peak / peak_a
        quantities['dist'] = distance / distance_a
        quantities['spd'] = distance / (op.end_time * 0.4)

    # Output mesh statistics and solver times
    quantities['mean_elements'] = av
    quantities['solver_timer'] = adapt_solve_timer
    quantities['adapt_solve_timer'] = adapt_solve_timer
    if op.mode == 'Tohoku':
        for g in op.gauges:
            quantities["TV "+g] = gauge_total_variation(quantities[g], gauge=g)

    return quantities


from thetis_adjoint import *
import pyadjoint
from fenics_adjoint.solving import SolveBlock                                       # For extracting adjoint solutions


def DWP(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    if op.plot_metric:
        mFile = File(op.directory() + "Metric2d.pvd")

    initTimer = clock()
    if op.plot_pvd:
        error_file = File(op.directory() + "ErrorIndicator2d.pvd")
        adjoint_file = File(op.directory() + "Adjoint2d.pvd")

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    V = op.mixed_space(mesh)
    q = Function(V)
    uv_2d, elev_2d = q.split()  # Needed to load data into
    uv_2d.rename('uv_2d')
    elev_2d.rename('elev_2d')
    P1 = FunctionSpace(mesh, "CG", 1)
    if op.mode == 'RossbyWave':    # Analytic final-time state
        peak_a, distance_a = peak_and_distance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
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
        options.horizontal_viscosity = diffusivity
        options.use_grad_div_viscosity_term = True                      # Symmetric viscous stress
        options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.timestep * op.timesteps_per_remesh
        options.simulation_end_time = op.end_time - 0.5 * op.timestep
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters = op.solver_parameters
        print("Using solver parameters %s" % options.timestepper_options.solver_parameters)
        options.timestep = op.timestep
        options.output_directory = op.directory()
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)
        primal_timer = clock()
        solver_obj.iterate()
        primal_timer = clock() - primal_timer
        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primal_timer)

        # Compute gradient
        gradient_timer = clock()
        compute_gradient(J, Control(b))     # TODO: Gradient w.r.t. some fields is more costly than others...
        gradient_timer = clock() - gradient_timer

        # Extract adjoint solutions
        dual_timer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.timesteps_per_export                            # Number of extra tape annotations in setup
        for i in range(N - 1, r - 1, -op.timesteps_per_export):
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            index_str = index_string(int((i - r) / op.timesteps_per_export))
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str, mode=FILE_CREATE) as sa:
                sa.store(dual_u)
                sa.store(dual_e)
                sa.close()
            if op.plot_pvd:
                adjoint_file.write(dual_u, dual_e, time=op.timestep * (i - r))
        dual_timer = clock() - dual_timer
        print('Dual run complete. Run time: %.3fs' % dual_timer)

    with pyadjoint.stop_annotating():

        error_timer = clock()
        for k in range(0, op.final_mesh_index()):  # Loop back over times to generate error estimators
            print('Generating error estimate %d / %d' % (k + 1, op.final_mesh_index()))
            with DumbCheckpoint(op.directory() + 'hdf5/Velocity2d_' + index_string(k), mode=FILE_READ) as lv:
                lv.load(uv_2d)
                lv.close()
            with DumbCheckpoint(op.directory() + 'hdf5/Elevation2d_' + index_string(k), mode=FILE_READ) as le:
                le.load(elev_2d)
                le.close()

            # Load adjoint data and form indicators
            epsilon.interpolate(inner(q, dual))
            for i in range(k, min(k + op.final_export() - op.first_export(), op.final_export())):
                with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_string(i), mode=FILE_READ) as la:
                    la.load(dual_u)
                    la.load(dual_e)
                    la.close()
                epsilon_.interpolate(inner(q, dual))
                epsilon = pointwise_max(epsilon, epsilon_)
            # epsilon = normalise_indicator(epsilon, op=op)
            epsilon.dat.data[:] = np.abs(epsilon.dat.data)
            with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_string(k), mode=FILE_CREATE) as se:
                se.store(epsilon)
                se.close()
            if op.plot_pvd:
                error_file.write(epsilon, time=float(k))
        error_timer = clock() - error_timer
        print('Errors estimated. Run time: %.3fs' % error_timer)

        # Run adaptive primal run
        cnt = 0
        adapt_solve_timer = 0.
        t = 0.
        q = Function(V)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        quantities = {}
        if op.mode == 'Tohoku':
            for g in op.gauges:
                quantities[g] = ()
        bdy = 200 if op.mode == 'Tohoku' else 'on_boundary'
        while cnt < op.final_index():
            adapt_timer = clock()
            for l in range(op.adaptations):                                  # TODO: Test this functionality

                # Construct metric
                index_str = index_string(int(cnt / op.timesteps_per_remesh))
                with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_str, mode=FILE_READ) as le:
                    le.load(epsilon)
                    le.close()
                estimate = Function(FunctionSpace(mesh, "CG", 1)).interpolate(interp(mesh, epsilon))
                print("#### DEBUG: error estimator norm = %.4e" % norm(estimate))
                M = isotropic_metric(estimate, invert=False, op=op)
                if op.gradate:
                    M_ = isotropic_metric(interp(mesh, H0), bdy=bdy, op=op)  # Initial boundary metric
                    M = metric_intersection(M, M_, bdy=bdy)
                    gradate_metric(M, op=op)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh

            if op.adaptations != 0 and op.plot_metric:
                M.rename('metric_2d')
                mFile.write(M, time=t)
            elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
            b, BCs, f, diffusivity = problem_domain(mesh=mesh, op=op)[3:]   # TODO: find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
            adapt_timer = clock() - adapt_timer

            # Solver object and equations
            adaptive_solver_obj = solver2d.FlowSolver2d(mesh, b)
            adaptive_options = adaptive_solver_obj.options
            adaptive_options.element_family = op.family
            adaptive_options.use_nonlinear_equations = True
            if diffusivity is not None:
                adaptive_options.horizontal_viscosity = interp(mesh, diffusivity)
            adaptive_options.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adaptive_options.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adaptive_options.simulation_export_time = op.timestep * op.timesteps_per_export
            adaptive_options.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adaptive_options.timestepper_type = op.timestepper
            adaptive_options.timestepper_options.solver_parameters = op.solver_parameters
            print("Using solver parameters %s" % adaptive_options.timestepper_options.solver_parameters)
            adaptive_options.timestep = op.timestep
            adaptive_options.output_directory = op.directory()
            if not op.plot_pvd:
                adaptive_options.no_exports = True
            adaptive_options.coriolis_frequency = f
            adaptive_solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
            adaptive_solver_obj.i_export = int(cnt / op.timesteps_per_export)
            adaptive_solver_obj.next_export_t = adaptive_solver_obj.i_export * adaptive_options.simulation_export_time
            adaptive_solver_obj.iteration = cnt
            adaptive_solver_obj.simulation_time = t
            for e in adaptive_solver_obj.exporters.values():
                e.set_next_export_ix(adaptive_solver_obj.i_export)

            # Evaluate callbacks and iterate
            cb1 = SWCallback(adaptive_solver_obj)
            cb1.op = op
            if cnt != 0:
                cb1.old_value = quantities['J_h']
            adaptive_solver_obj.add_callback(cb1, 'timestep')
            if op.mode == 'Tohoku':
                cb2 = callback.DetectorsCallback(adaptive_solver_obj,
                                                 [op.gauge_coordinates(g) for g in op.gauges],
                                                 ['elev_2d'],
                                                 'timeseries',
                                                 op.gauges,
                                                 export_to_hdf5=True)
                adaptive_solver_obj.add_callback(cb2, 'timestep')
            adaptive_solver_obj.bnd_functions['shallow_water'] = BCs
            solver_timer = clock()
            adaptive_solver_obj.iterate()
            solver_timer = clock() - solver_timer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
            if op.mode == 'Tohoku':
                extract_gauge_data(quantities, op=op)

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adapt_timer, solver_timer, nEle, Sn, mM, cnt * op.timestep)
            adapt_solve_timer += adapt_timer + solver_timer

            # Extract fields for next solver block
            uv_2d, elev_2d = adaptive_solver_obj.fields.solution_2d.split()

            # Measure error using metrics, as in Huang et al.
        if op.mode == 'RossbyWave':
            peak, distance = peak_and_distance(elev_2d, op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.end_time * 0.4)

        # Output mesh statistics and solver times
        total_timer = error_timer + adapt_solve_timer
        if not regen:
            total_timer += primal_timer + gradient_timer + dual_timer
        quantities['mean_elements'] = av
        quantities['solver_timer'] = total_timer
        quantities['adapt_solve_timer'] = adapt_solve_timer
        if op.mode == 'Tohoku':
            for g in op.gauges:
                quantities["TV " + g] = gauge_total_variation(quantities[g], gauge=g)

        return quantities


def DWR(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    if op.plot_metric:
        mFile = File(op.directory() + "Metric2d.pvd")

    initTimer = clock()
    if op.plot_pvd:
        residual_file = File(op.directory() + "Residual2d.pvd")
        error_file = File(op.directory() + "ErrorIndicator2d.pvd")
        adjoint_file = File(op.directory() + "Adjoint2d.pvd")

    # Initialise domain and physical parameters
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    V = op.mixed_space(mesh)
    q = Function(V)
    uv_2d, elev_2d = q.split()    # Needed to load data into
    uv_2d.rename('uv_2d')
    elev_2d.rename('elev_2d')
    P0 = FunctionSpace(mesh, "DG", 0)
    P1 = FunctionSpace(mesh, "CG", 1)
    if op.mode == 'RossbyWave':    # Analytic final-time state
        peak_a, distance_a = peak_and_distance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename('Adjoint velocity')
    dual_e.rename('Adjoint elevation')
    epsilon = Function(P1, name='error_2d')
    residual_2d = Function(P0)

    if op.order_increase:
        duale = Function(op.mixed_space(mesh, enrich=True))
        duale_u, duale_e = duale.split()

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
        options.horizontal_viscosity = diffusivity
        options.use_grad_div_viscosity_term = True                      # Symmetric viscous stress
        options.use_lax_friedrichs_velocity = False                     # TODO: This is a temporary fix
        options.coriolis_frequency = f
        options.simulation_export_time = op.timestep * op.timesteps_per_export
        options.simulation_end_time = op.end_time - 0.5 * op.timestep
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters_tracer = op.solver_parameters
        print("Using solver parameters %s" % options.timestepper_options.solver_parameters)
        options.timestep = op.timestep
        options.output_directory = op.directory()   # Need this for residual callback
        options.export_diagnostics = False
        # options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        cb2 = callback.ExplicitErrorCallback(solver_obj, export_to_hdf5=True)
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.add_callback(cb2, 'export')
        solver_obj.bnd_functions['shallow_water'] = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)

        primal_timer = clock()
        solver_obj.iterate()
        primal_timer = clock() - primal_timer
        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primal_timer)

        # Compute gradient
        gradient_timer = clock()
        compute_gradient(J, Control(b))     # TODO: Gradient w.r.t. some fields is more costly than others...
        gradient_timer = clock() - gradient_timer

        # Extract adjoint solutions
        dual_timer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.timesteps_per_remesh                       # Number of extra tape annotations in setup
        for i in range(r, N, op.timesteps_per_remesh):        # Iterate r is the first timestep
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            index_str = index_string(int((i - r) / op.timesteps_per_remesh))
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + index_str,  mode=FILE_CREATE) as sa:
                sa.store(dual_u)
                sa.store(dual_e)
                sa.close()
            if op.plot_pvd:
                adjoint_file.write(dual_u, dual_e, time=op.timestep * (i - r))
        dual_timer = clock() - dual_timer
        print('Dual run complete. Run time: %.3fs' % dual_timer)

        with pyadjoint.stop_annotating():

            residuals = []
            error_timer = clock()
            for k in range(0, int(op.final_index() / op.timesteps_per_export)):
                print('Generating error estimate %d / %d'
                      % (int(k/op.exports_per_remesh()) + 1, int(op.final_index() / op.timesteps_per_remesh)))

                # Load residuals
                with DumbCheckpoint(op.directory() + 'hdf5/ExplicitError2d_' + index_string(k), mode=FILE_READ) as lr:
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
                        la.load(dual_u)
                        la.load(dual_e)
                        la.close()
                    if op.order_increase:
                        duale_u.interpolate(dual_u)
                        duale_e.interpolate(dual_e)
                        raise NotImplementedError   # TODO: Requires patchwise interpolation
                    else:
                        # TODO: Get adjoint from previous (/next?) step
                        epsilon.interpolate(difference_quotient_estimator(solver_obj, residual_2d, dual, dual)) # Note: Would be subtract with no L-inf
                    # epsilon = normalise_indicator(epsilon, op=op)
                    epsilon.dat.data[:] = np.abs(epsilon.dat.data)
                    epsilon.rename("Error indicator")
                    with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_str, mode=FILE_CREATE) as se:
                        se.store(epsilon)
                        se.close()
                    if op.plot_pvd:
                        error_file.write(epsilon, time=float(op.timestep * op.timesteps_per_remesh * k))
            error_timer = clock() - error_timer
            print('Errors estimated. Run time: %.3fs' % error_timer)

    with pyadjoint.stop_annotating():

        # Run adaptive primal run
        cnt = 0
        adapt_solve_timer = 0.
        t = 0.
        q = Function(V)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        quantities = {}
        if op.mode == 'Tohoku':
            for g in op.gauges:
                quantities[g] = ()
        bdy = 200 if op.mode == 'Tohoku' else 'on_boundary'
        # bdy = 'on_boundary'
        while cnt < op.final_index():
            adapt_timer = clock()
            for l in range(op.adaptations):                          # TODO: Test this functionality

                # Construct metric
                index_str = index_string(int(cnt / op.timesteps_per_remesh))
                with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + index_str, mode=FILE_READ) as le:
                    le.load(epsilon)
                    le.close()
                estimate = Function(FunctionSpace(mesh, "CG", 1)).assign(interp(mesh, epsilon))
                print("#### DEBUG: error estimator norm = %.4e" % norm(estimate))
                M = isotropic_metric(estimate, invert=False, op=op)
                if op.gradate:
                    # br = Function(P1).interpolate(boundary_region(mesh, 200, 5e8))
                    # ass = assemble(interp(mesh, H0) * br / assemble(100 * br * dx))
                    # File('plots/tohoku/boundary_region.pvd').write(ass)
                    # M_ = isotropic_metric(ass, op=op)
                    # M = metric_intersection(M, M_)

                    M_ = isotropic_metric(interp(mesh, H0), bdy=bdy, op=op)   # Initial boundary metric
                    M = metric_intersection(M, M_, bdy=bdy)
                    M = gradate_metric(M, op=op)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh

            if op.adaptations != 0 and op.plot_metric:
                M.rename('metric_2d')
                mFile.write(M, time=t)
            elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
            b, BCs, f, diffusivity = problem_domain(mesh=mesh, op=op)[3:]   # TODO: Find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
            adapt_timer = clock() - adapt_timer

            # Solver object and equations
            adaptive_solver_obj = solver2d.FlowSolver2d(mesh, b)
            adaptive_options = adaptive_solver_obj.options
            adaptive_options.element_family = op.family
            adaptive_options.use_nonlinear_equations = True
            if diffusivity is not None:
                adaptive_options.horizontal_viscosity = interp(mesh, diffusivity)
            adaptive_options.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adaptive_options.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adaptive_options.simulation_export_time = op.timestep * op.timesteps_per_export
            adaptive_options.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adaptive_options.timestepper_type = op.timestepper
            adaptive_options.timestepper_options.solver_parameters = op.solver_parameters
            print("Using solver parameters %s" % adaptive_options.timestepper_options.solver_parameters)
            adaptive_options.timestep = op.timestep
            adaptive_options.output_directory = op.directory()
            if not op.plot_pvd:
                adaptive_options.no_exports = True
            adaptive_options.coriolis_frequency = f
            adaptive_solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
            adaptive_solver_obj.i_export = int(cnt / op.timesteps_per_export)
            adaptive_solver_obj.next_export_t = adaptive_solver_obj.i_export * adaptive_options.simulation_export_time
            adaptive_solver_obj.iteration = cnt
            adaptive_solver_obj.simulation_time = t
            for e in adaptive_solver_obj.exporters.values():
                e.set_next_export_ix(adaptive_solver_obj.i_export)

            # Evaluate callbacks and iterate
            cb1 = SWCallback(adaptive_solver_obj)
            cb1.op = op
            if cnt != 0:
                cb1.old_value = quantities['J_h']
            adaptive_solver_obj.add_callback(cb1, 'timestep')
            if op.mode == 'Tohoku':
                cb2 = callback.DetectorsCallback(adaptive_solver_obj,
                                                 [op.gauge_coordinates(g) for g in op.gauges],
                                                 ['elev_2d'],
                                                 'timeseries',
                                                 op.gauges,
                                                 export_to_hdf5=True)
                adaptive_solver_obj.add_callback(cb2, 'timestep')
            adaptive_solver_obj.bnd_functions['shallow_water'] = BCs
            solver_timer = clock()
            adaptive_solver_obj.iterate()
            solver_timer = clock() - solver_timer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
            if op.mode == 'Tohoku':
                extract_gauge_data(quantities, op=op)

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adapt_timer, solver_timer, nEle, Sn, mM, cnt * op.timestep)
            adapt_solve_timer += adapt_timer + solver_timer

            # Extract fields for next solver block
            uv_2d, elev_2d = adaptive_solver_obj.fields.solution_2d.split()

            # Measure error using metrics, as in Huang et al.
        if op.mode == 'RossbyWave':
            peak, distance = peak_and_distance(elev_2d, op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.end_time * 0.4)

            # Output mesh statistics and solver times
        total_timer = error_timer + adapt_solve_timer
        if not regen:
            total_timer += primal_timer + gradient_timer + dual_timer
        quantities['mean_elements'] = av
        quantities['solver_timer'] = total_timer
        quantities['adapt_solve_timer'] = adapt_solve_timer
        if op.mode == 'Tohoku':
            for g in op.gauges:
                quantities["TV " + g] = gauge_total_variation(quantities[g], gauge=g)

        return quantities


def tsunami(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    solvers = {'FixedMesh': FixedMesh, 'HessianBased': HessianBased, 'DWP': DWP, 'DWR': DWR}

    return solvers[op.approach](mesh, u0, eta0, b, BCs, f, diffusivity, regen=regen, op=op)
