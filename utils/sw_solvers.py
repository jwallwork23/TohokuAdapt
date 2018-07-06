from thetis import *
from thetis.callback import DetectorsCallback

import numpy as np
from time import clock
import h5py

from utils.adaptivity import *
from utils.callbacks import SWCallback, ObjectiveSWCallback
from utils.error_
from utils.interpolation import interp, mixedPairInterp
from utils.misc import indexString, peakAndDistance, bdyRegion
from utils.setup import problemDomain, RossbyWaveSolution
from utils.timeseries import gaugeTV


__all__ = ["tsunami"]


def fixedMesh(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')

    # Initialise domain and physical parameters
    try:
        assert float(physical_constants['g_grav'].dat.data) == op.g
    except:
        physical_constants['g_grav'].assign(op.g)
    V = op.mixed_space(mesh)                               # TODO: Parallelise this (and below)
    if op.mode == 'rossby-wave':            # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Initialise solver
    gauges = ["P02", "P06"]
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
    if op.mode == 'tohoku':
        cb2 = DetectorsCallback(solver_obj,
                                [op.gauge_coordinates(g) for g in gauges],
                                ['elev_2d'],
                                'timeseries',
                                gauges,
                                export_to_hdf5=True)
        solver_obj.add_callback(cb2, 'timestep')
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions['shallow_water'] = BCs

    # Solve and extract timeseries / functionals
    quantities = {}
    solverTimer = clock()
    solver_obj.iterate()
    solverTimer = clock() - solverTimer
    quantities['J_h'] = cb1.get_val()          # Evaluate objective functional
    if op.mode == 'tohoku':
        hf = h5py.File(op.directory() + 'diagnostic_timeseries.hdf5', 'r')
        for g in gauges:
            quantities[g] = np.array(hf.get(g))
        hf.close()

    # Measure error using metrics, as in Huang et al.     # TODO: Parallelise this (and above)
    if op.mode == 'rossby-wave':
        peak, distance = peakAndDistance(solver_obj.fields.solution_2d.split()[1], op=op)
        distance += 48. # Account for periodic domain
        quantities['peak'] = peak/peak_a
        quantities['dist'] = distance/distance_a
        quantities['spd'] = distance /(op.end_time * 0.4)

    # Output mesh statistics and solver times
    quantities['meanElements'] = mesh.num_cells()
    quantities['solverTimer'] = solverTimer
    quantities['adaptSolveTimer'] = 0.
    for g in gauges:
        quantities["TV "+g] = gaugeTV(quantities[g], gauge=g)

    return quantities


def hessianBased(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
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
    if op.mode == 'rossby-wave':    # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling   # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    cnt = 0
    t = 0.

    adaptSolveTimer = 0.
    quantities = {}
    gauges = ["P02", "P06"]
    for g in gauges:
        quantities[g] = ()
    while cnt < op.final_index():
        adaptTimer = clock()
        P1 = FunctionSpace(mesh, "CG", 1)

        if op.adapt_field != 's':
            height = Function(P1).interpolate(elev_2d)
        if op.adapt_field != 'f':
            spd = Function(P1).interpolate(sqrt(dot(uv_2d, uv_2d)))
        for l in range(op.adaptations):                  # TODO: Test this functionality

            # Construct metric
            if op.adapt_field != 's':
                M = steadyMetric(height, op=op)
            if op.adapt_field != 'f' and cnt != 0:   # Can't adapt to zero velocity
                M2 = steadyMetric(spd, op=op)
                if op.adapt_field != 'b':
                    M = M2
                else:
                    try:
                        M = metricIntersection(M, M2)
                    except:
                        print("WARNING: null fluid speed metric")
                        M = metricIntersection(M2, M)
            if op.adapt_on_bathymetry and not (op.adapt_field != 'f' and cnt == 0):
                M2 = steadyMetric(b, op=op)
                M = M2 if op.adapt_field != 'f' and cnt == 0. else metricIntersection(M, M2)     # TODO: Convex combination?

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
            b, BCs, f, diffusivity = problemDomain(mesh=mesh, op=op)[3:]     # TODO: find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
        adaptTimer = clock() - adaptTimer

        # Solver object and equations
        adapSolver = solver2d.FlowSolver2d(mesh, b)
        adapOpt = adapSolver.options
        adapOpt.element_family = op.family
        adapOpt.use_nonlinear_equations = True
        if diffusivity is not None:
            adapOpt.horizontal_viscosity = interp(mesh, diffusivity)
        adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
        adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
        adapOpt.simulation_export_time = op.timestep * op.timesteps_per_export
        adapOpt.simulation_end_time = t + op.timestep * (op.timesteps_per_remesh - 0.5)
        adapOpt.timestepper_type = op.timestepper
        adapOpt.timestepper_options.solver_parameters = op.solver_parameters
        print("Using solver parameters %s" % adapOpt.timestepper_options.solver_parameters)
        adapOpt.timestep = op.timestep
        adapOpt.output_directory = op.directory()
        if not op.plot_pvd:
            adapOpt.no_exports = True
        adapOpt.coriolis_frequency = f
        adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
        adapSolver.i_export = int(cnt / op.timesteps_per_export)
        adapSolver.next_export_t = adapSolver.i_export * adapSolver.options.simulation_export_time
        adapSolver.iteration = cnt
        adapSolver.simulation_time = t
        for e in adapSolver.exporters.values():
            e.set_next_export_ix(adapSolver.i_export)

        # Establish callbacks and iterate
        cb1 = SWCallback(adapSolver)
        cb1.op = op
        if cnt != 0:
            cb1.old_value = quantities['J_h']
        adapSolver.add_callback(cb1, 'timestep')
        if op.mode == 'tohoku':
            cb2 = DetectorsCallback(adapSolver,
                                    [op.gauge_coordinates(g) for g in gauges],
                                    ['elev_2d'],
                                    'timeseries',
                                    gauges,
                                    export_to_hdf5=True)
            adapSolver.add_callback(cb2, 'timestep')
        adapSolver.bnd_functions['shallow_water'] = BCs
        solverTimer = clock()
        adapSolver.iterate()
        solverTimer = clock() - solverTimer
        quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
        if op.mode == 'tohoku':
            hf = h5py.File(op.directory() + 'diagnostic_timeseries.hdf5', 'r')
            for g in gauges:
                quantities[g] += tuple(hf.get(g))
            hf.close()

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

    # Measure error using metrics, as in Huang et al.
    if op.mode == 'rossby-wave':
        peak, distance = peakAndDistance(elev_2d, op=op)
        quantities['peak'] = peak / peak_a
        quantities['dist'] = distance / distance_a
        quantities['spd'] = distance / (op.end_time * 0.4)

    # Output mesh statistics and solver times
    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['adaptSolveTimer'] = adaptSolveTimer
    for g in gauges:
        quantities["TV "+g] = gaugeTV(quantities[g], gauge=g)

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
        errorFile = File(op.directory() + "ErrorIndicator2d.pvd")
        adjointFile = File(op.directory() + "Adjoint2d.pvd")

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
    if op.mode == 'rossby-wave':    # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

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
        primalTimer = clock()
        solver_obj.iterate()
        primalTimer = clock() - primalTimer
        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        compute_gradient(J, Control(b))
        gradientTimer = clock() - gradientTimer

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.timesteps_per_export                            # Number of extra tape annotations in setup
        for i in range(N - 1, r - 1, -op.timesteps_per_export):
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + indexString(int((i - r) / op.timesteps_per_export)), mode=FILE_CREATE) as saveAdj:
                saveAdj.store(dual_u)
                saveAdj.store(dual_e)
                saveAdj.close()
            if op.plot_pvd:
                adjointFile.write(dual_u, dual_e, time=op.timestep * (i - r))
        dualTimer = clock() - dualTimer
        print('Dual run complete. Run time: %.3fs' % dualTimer)

    with pyadjoint.stop_annotating():

        errorTimer = clock()
        for k in range(0, op.final_mesh_index()):  # Loop back over times to generate error estimators
            print('Generating error estimate %d / %d' % (k + 1, op.final_mesh_index()))
            with DumbCheckpoint(op.directory() + 'hdf5/Velocity2d_' + indexString(k), mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d)
                loadVel.close()
            with DumbCheckpoint(op.directory() + 'hdf5/Elevation2d_' + indexString(k), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d)
                loadElev.close()

            # Load adjoint data and form indicators
            epsilon.interpolate(inner(q, dual))
            for i in range(k, min(k + op.final_export() - op.first_export(), op.final_export())):
                with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + indexString(i), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_u)
                    loadAdj.load(dual_e)
                    loadAdj.close()
                epsilon_.interpolate(inner(q, dual))
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
        q = Function(V)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        quantities = {}
        gauges = ["P02", "P06"]
        for g in gauges:
            quantities[g] = ()
        bdy = 200 if op.mode == 'tohoku' else 'on_boundary'
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
                    # metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh

            if op.adaptations != 0 and op.plot_metric:
                M.rename('metric_2d')
                mFile.write(M, time=t)
            elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
            b, BCs, f, diffusivity = problemDomain(mesh=mesh, op=op)[3:]             # TODO: find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            if diffusivity is not None:
                adapOpt.horizontal_viscosity = interp(mesh, diffusivity)
            adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.timestep * op.timesteps_per_export
            adapOpt.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestepper_options.solver_parameters = op.solver_parameters
            print("Using solver parameters %s" % adapOpt.timestepper_options.solver_parameters)
            adapOpt.timestep = op.timestep
            adapOpt.output_directory = op.directory()
            if not op.plot_pvd:
                adapOpt.no_exports = True
            adapOpt.coriolis_frequency = f
            adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
            adapSolver.i_export = int(cnt / op.timesteps_per_export)
            adapSolver.next_export_t = adapSolver.i_export * adapSolver.options.simulation_export_time
            adapSolver.iteration = cnt
            adapSolver.simulation_time = t
            for e in adapSolver.exporters.values():
                e.set_next_export_ix(adapSolver.i_export)

            # Evaluate callbacks and iterate
            cb1 = SWCallback(adapSolver)
            cb1.op = op
            if cnt != 0:
                cb1.old_value = quantities['J_h']
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode == 'tohoku':
                cb2 = DetectorsCallback(adapSolver,
                                        [op.gauge_coordinates(g) for g in gauges],
                                        ['elev_2d'],
                                        'timeseries',
                                        gauges,
                                        export_to_hdf5=True)
                adapSolver.add_callback(cb2, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
            if op.mode == 'tohoku':
                hf = h5py.File(op.directory() + 'diagnostic_timeseries.hdf5', 'r')
                for g in gauges:
                    quantities[g] += tuple(hf.get(g))
                hf.close()

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.timestep)
            adaptSolveTimer += adaptTimer + solverTimer

            # Extract fields for next solver block
            uv_2d, elev_2d = adapSolver.fields.solution_2d.split()

            # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            peak, distance = peakAndDistance(elev_2d, op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.end_time * 0.4)

        # Output mesh statistics and solver times
        totalTimer = errorTimer + adaptSolveTimer
        if not regen:
            totalTimer += primalTimer + gradientTimer + dualTimer
        quantities['meanElements'] = av
        quantities['solverTimer'] = totalTimer
        quantities['adaptSolveTimer'] = adaptSolveTimer
        for g in gauges:
            quantities["TV " + g] = gaugeTV(quantities[g], gauge=g)

        return quantities


def DWR(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):     # TODO: Store optimal mesh, 'intersected' over all rm steps
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    if op.plot_metric:
        mFile = File(op.directory() + "Metric2d.pvd")

    initTimer = clock()
    if op.plot_pvd:
        residualFile = File(op.directory() + "Residual2d.pvd")
        errorFile = File(op.directory() + "ErrorIndicator2d.pvd")
        adjointFile = File(op.directory() + "Adjoint2d.pvd")

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
    P1 = FunctionSpace(mesh, "CG", 1)
    if op.mode == 'rossby-wave':    # Analytic final-time state
        peak_a, distance_a = peakAndDistance(RossbyWaveSolution(V, op=op).__call__(t=op.end_time).split()[1])

    # Define Functions relating to a posteriori DWR error estimator
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")

    if op.order_increase:
        Ve = op.mixed_space(mesh, enrich=True)
        duale = Function(Ve)
        duale_u, duale_e = duale.split()
        epsilon = Function(P1, name='error_2d')
    else:                                   # Copy standard variables to mimic enriched space labels
        Ve = V
        epsilon = Function(P1, name='error_2d')
    v = TestFunction(FunctionSpace(mesh, "DG", 0)) # For forming error indicators
    rho = Function(Ve)
    rho_u, rho_e = rho.split()
    rho_u.rename("Momentum error")
    rho_e.rename("Continuity error")
    brho = Function(Ve, name="Boundary error")
    brho_u, brho_e = brho.split()
    temp = Function(Ve)
    temp_u, temp_e = temp.split()
    btemp = Function(Ve)
    btemp_u, btemp_e = btemp.split()

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.target_vertices = mesh.num_vertices() * op.rescaling  # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    t = 0.

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
        options.simulation_end_time = (op.timesteps_per_export - 0.5) * op.timestep
        options.timestepper_type = op.timestepper
        options.timestepper_options.solver_parameters_tracer = op.solver_parameters
        print("Using solver parameters %s" % options.timestepper_options.solver_parameters)
        options.timestep = op.timestep
        options.output_directory = op.directory()   # Need this for residual callback
        options.export_diagnostics = True
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']            # TODO: EXPORT FROM PREVIOUS STEP?
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        cb1 = ObjectiveSWCallback(solver_obj)
        cb1.op = op
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        initTimer = clock() - initTimer
        print('Problem initialised. Setup time: %.3fs' % initTimer)

        cnt = 0
        primalTimer = 0.
        while solver_obj.simulation_time < op.end_time - 0.5 * op.timestep:

            with pyadjoint.stop_annotating():
                uv_old, elev_old = solver_obj.timestepper.solution_old.split()
                uv_old.rename("Previous velocity")
                elev_old.rename("Previous elevation")
                with DumbCheckpoint(op.directory() + 'hdf5/Previous2d_' + indexString(cnt), mode=FILE_CREATE) as savePrev:
                    savePrev.store(uv_old)
                    savePrev.store(elev_old)
                    savePrev.close()

                if cnt != 0:
                    solver_obj.load_state(cnt, iteration=cnt*op.timesteps_per_export)

            # Run simulation
            stepTimer = clock()
            solver_obj.iterate()
            stepTimer = clock() - stepTimer
            primalTimer += stepTimer
            cnt += 1
            t += op.timestep * op.timesteps_per_export
            options.simulation_end_time = t + (op.timesteps_per_export - 0.5) * op.timestep

        J = cb1.get_val()                        # Assemble objective functional for adjoint computation
        print('Primal run complete. Solver time: %.3fs' % primalTimer)

        # Compute gradient
        gradientTimer = clock()
        compute_gradient(J, Control(b))
        gradientTimer = clock() - gradientTimer

        # Extract adjoint solutions
        dualTimer = clock()
        tape = get_working_tape()
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        N = len(solve_blocks)
        r = N % op.timesteps_per_remesh                       # Number of extra tape annotations in setup
        for i in range(r, N, op.timesteps_per_remesh):        # Iterate r is the first timestep
            dual.assign(solve_blocks[i].adj_sol)
            dual_u, dual_e = dual.split()
            with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + indexString(int((i - r) / op.timesteps_per_remesh)),  mode=FILE_CREATE) as saveAdj:
                saveAdj.store(dual_u)
                saveAdj.store(dual_e)
                saveAdj.close()
            if op.plot_pvd:
                adjointFile.write(dual_u, dual_e, time=op.timestep * (i - r))
        dualTimer = clock() - dualTimer
        print('Dual run complete. Run time: %.3fs' % dualTimer)

        with pyadjoint.stop_annotating():

            residuals = {'Velocity': [], 'Elevation': [], 'Boundary': []}
            errorTimer = clock()
            for k in range(0, int(op.final_index() / op.timesteps_per_export)):
                print('Generating error estimate %d / %d' % (int(k/op.exports_per_remesh()) + 1, int(op.final_index() / op.timesteps_per_remesh)))

                # Generate residuals
                with DumbCheckpoint(op.directory() + 'hdf5/Velocity2d_' + indexString(k), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name="uv_2d")
                    loadVel.close()
                with DumbCheckpoint(op.directory() + 'hdf5/Elevation2d_' + indexString(k), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name="elev_2d")
                    loadElev.close()
                with DumbCheckpoint(op.directory() + 'hdf5/Previous2d_' + indexString(k), mode=FILE_READ) as loadPrev:
                    loadPrev.load(uv_old, name="Previous velocity")
                    loadPrev.load(elev_old, name="Previous elevation")
                    loadPrev.close()
                tic = clock()
                err_u, err_e, bres_u, bres_e = sw_strong_residual(solver_obj, uv_2d, elev_2d, uv_old, elev_old, Ve, op=op)
                print("Residual computation: %.2fs" % (clock()- tic))
                residuals['Velocity'].append(err_u)
                residuals['Elevation'].append(err_e)
                residuals['bdyVelocity'].append(bres_u)
                residuals['bdyElevation'].append(bres_e)
                if k % op.exports_per_remesh() == op.exports_per_remesh()-1:

                    # L-inf     # TODO: This is grossly inefficient
                    rho_u.interpolate(residuals['Velocity'][0])
                    rho_e.interpolate(residuals['Elevation'][0])
                    brho_u.interpolate(residuals['bdyVelocity'][0])
                    brho_e.interpolate(residuals['bdyElevation'][0])
                    rho_u.dat.data[:] = np.abs(rho_u.dat.data)
                    rho_e.dat.data[:] = np.abs(rho_e.dat.data)
                    brho_u.dat.data[:] = np.abs(brho_u.dat.data)
                    brho_e.dat.data[:] = np.abs(brho_e.dat.data)
                    for i in range(1, len(residuals['Velocity'])):
                        temp_u.interpolate(residuals['Velocity'][i])
                        temp_e.interpolate(residuals['Elevation'][i])
                        btemp_u.interpolate(residuals['bdyVelocity'][i])
                        btemp_e.interpolate(residuals['bdyElevation'][i])
                        temp_u.dat.data[:] = np.abs(temp_u.dat.data)
                        temp_e.dat.data[:] = np.abs(temp_e.dat.data)
                        btemp_u.dat.data[:] = np.abs(btemp_u.dat.data)
                        btemp_e.dat.data[:] = np.abs(btemp_e.dat.data)
                        for j in range(len(temp_e.dat.data)):
                            rho_u.dat.data[j, 0] = max(temp_u.dat.data[j, 0], rho_u.dat.data[j, 0])
                            rho_u.dat.data[j, 1] = max(temp_u.dat.data[j, 1], rho_u.dat.data[j, 1])
                            rho_e.dat.data[j] = max(temp_e.dat.data[j], rho_e.dat.data[j])
                            brho_u.dat.data[j, 0] = max(btemp_u.dat.data[j, 0], brho_u.dat.data[j, 0])
                            brho_u.dat.data[j, 1] = max(btemp_u.dat.data[j, 1], brho_u.dat.data[j, 1])
                            brho_e.dat.data[j] = max(btemp_e.dat.data[j], brho_e.dat.data[j])

                    # # L1
                    # err_u = op.timestep * sum(abs(residuals['Velocity'][i] + residuals['Velocity'][i - 1]) for i in range(1, op.exports_per_remesh()))
                    # err_e = op.timestep * sum(abs(residuals['Elevation'][i] + residuals['Elevation'][i - 1]) for i in range(1, op.exports_per_remesh()))

                    # # Time integrate residual over current 'window'
                    # err_u = op.timestep * sum(residuals['Velocity'][i] + residuals['Velocity'][i-1] for i in range(1, op.exports_per_remesh()))
                    # err_e = op.timestep * sum(residuals['Elevation'][i] + residuals['Elevation'][i-1] for i in range(1, op.exports_per_remesh()))
                    #
                    # rho_u.interpolate(err_u)
                    # rho_e.interpolate(err_e)

                    residuals = {'Velocity': [], 'Elevation': [], 'bdyVelocity': [], 'bdyElevation': []}
                    if op.plot_pvd:
                        residualFile.write(rho_u, rho_e, time=float(op.timestep * op.timesteps_per_remesh * (k+1)))

                    # Load adjoint data and form indicators
                    indexStr = indexString(int((k+1)/op.exports_per_remesh()-1))
                    with DumbCheckpoint(op.directory() + 'hdf5/Adjoint2d_' + indexStr, mode=FILE_READ) as loadAdj:
                        loadAdj.load(dual_u)
                        loadAdj.load(dual_e)
                        loadAdj.close()
                    if op.order_increase:                  # TODO: Replace adj with difference
                        duale_u.interpolate(dual_u)     # TODO: ... between higher order adj and adj on comp. mesh.
                        duale_e.interpolate(dual_e)     # TODO: ... h.o. interpolation should be patchwise.
                        epsilon.interpolate(assemble(v * (inner(rho, duale) + inner(brho, duale)) * dx))
                                                                          # ^ Would be subtract with no L-inf
                        # TODO: Also include method of difference quotients
                    else:
                        epsilon.interpolate(assemble(v * (inner(rho, dual) + inner(brho, dual)) * dx))
                    epsilon = normaliseIndicator(epsilon, op=op)         # ^ Would be subtract with no L-inf
                    epsilon.rename("Error indicator")
                    with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + indexStr, mode=FILE_CREATE) as saveErr:
                        saveErr.store(epsilon)
                        saveErr.close()
                    if op.plot_pvd:
                        errorFile.write(epsilon, time=float(op.timestep * op.timesteps_per_remesh * k))
            errorTimer = clock() - errorTimer
            print('Errors estimated. Run time: %.3fs' % errorTimer)

    with pyadjoint.stop_annotating():

        # Run adaptive primal run
        cnt = 0
        adaptSolveTimer = 0.
        t = 0.
        q = Function(V)
        uv_2d, elev_2d = q.split()
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)
        quantities = {}
        gauges = ["P02", "P06"]
        for g in gauges:
            quantities[g] = ()
        bdy = 200 if op.mode == 'tohoku' else 'on_boundary'
        # bdy = 'on_boundary'
        while cnt < op.final_index():
            adaptTimer = clock()
            for l in range(op.adaptations):                          # TODO: Test this functionality

                # Construct metric
                indexStr = indexString(int(cnt / op.timesteps_per_remesh))
                with DumbCheckpoint(op.directory() + 'hdf5/ErrorIndicator2d_' + indexStr, mode=FILE_READ) as loadErr:
                    loadErr.load(epsilon)
                    loadErr.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).assign(interp(mesh, epsilon))
                M = isotropicMetric(errEst, invert=False, op=op)
                if op.gradate:
                    # br = Function(P1).interpolate(bdyRegion(mesh, 200, 5e8))
                    # ass = assemble(interp(mesh, H0) * br / assemble(100 * br * dx))
                    # File('plots/tohoku/bdyRegion.pvd').write(ass)
                    # M_ = isotropicMetric(ass, op=op)
                    # M = metricIntersection(M, M_)

                    M_ = isotropicMetric(interp(mesh, H0), bdy=bdy, op=op)   # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=bdy)
                    M = metricGradation(M, op=op)

                # Adapt mesh and interpolate variables
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                # File('plots/tohoku/mesh.pvd').write(mesh.coordinates)
                # exit(0)

            if op.adaptations != 0 and op.plot_metric:
                M.rename('metric_2d')
                mFile.write(M, time=t)
            elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
            b, BCs, f, diffusivity = problemDomain(mesh=mesh, op=op)[3:]           # TODO: Find a different way to reset these
            uv_2d.rename('uv_2d')
            elev_2d.rename('elev_2d')
            adaptTimer = clock() - adaptTimer

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True
            if diffusivity is not None:
                adapOpt.horizontal_viscosity = interp(mesh, diffusivity)
            adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
            adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.timestep * op.timesteps_per_export
            adapOpt.simulation_end_time = t + (op.timesteps_per_remesh - 0.5) * op.timestep
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestepper_options.solver_parameters = op.solver_parameters
            print("Using solver parameters %s" % adapOpt.timestepper_options.solver_parameters)
            adapOpt.timestep = op.timestep
            adapOpt.output_directory = op.directory()
            if not op.plot_pvd:
                adapOpt.no_exports = True
            adapOpt.coriolis_frequency = f
            adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
            adapSolver.i_export = int(cnt / op.timesteps_per_export)
            adapSolver.next_export_t = adapSolver.i_export * adapSolver.options.simulation_export_time
            adapSolver.iteration = cnt
            adapSolver.simulation_time = t
            for e in adapSolver.exporters.values():
                e.set_next_export_ix(adapSolver.i_export)

            # Evaluate callbacks and iterate
            cb1 = SWCallback(adapSolver)
            cb1.op = op
            if cnt != 0:
                cb1.old_value = quantities['J_h']
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode == 'tohoku':
                cb2 = DetectorsCallback(adapSolver,
                                        [op.gauge_coordinates(g) for g in gauges],
                                        ['elev_2d'],
                                        'timeseries',
                                        gauges,
                                        export_to_hdf5=True)
                adapSolver.add_callback(cb2, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            solverTimer = clock()
            adapSolver.iterate()
            solverTimer = clock() - solverTimer
            quantities['J_h'] = cb1.get_val()  # Evaluate objective functional
            if op.mode == 'tohoku':
                hf = h5py.File(op.directory() + 'diagnostic_timeseries.hdf5', 'r')
                for g in gauges:
                    quantities[g] += tuple(hf.get(g))
                hf.close()

            # Get mesh stats
            nEle = mesh.num_cells()
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.timesteps_per_remesh
            t += op.timesteps_per_remesh * op.timestep
            av = op.adaptation_stats(int(cnt / op.timesteps_per_remesh + 1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.timestep)
            adaptSolveTimer += adaptTimer + solverTimer

            # Extract fields for next solver block
            uv_2d, elev_2d = adapSolver.fields.solution_2d.split()

            # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            peak, distance = peakAndDistance(elev_2d, op=op)
            quantities['peak'] = peak / peak_a
            quantities['dist'] = distance / distance_a
            quantities['spd'] = distance / (op.end_time * 0.4)

            # Output mesh statistics and solver times
        totalTimer = errorTimer + adaptSolveTimer
        if not regen:
            totalTimer += primalTimer + gradientTimer + dualTimer
        quantities['meanElements'] = av
        quantities['solverTimer'] = totalTimer
        quantities['adaptSolveTimer'] = adaptSolveTimer
        for g in gauges:
            quantities["TV " + g] = gaugeTV(quantities[g], gauge=g)

        return quantities


def tsunami(mesh, u0, eta0, b, BCs={}, f=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    solvers = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased, 'DWP': DWP, 'DWR': DWR}

    return solvers[op.approach](mesh, u0, eta0, b, BCs, f, diffusivity, regen=regen, op=op)
