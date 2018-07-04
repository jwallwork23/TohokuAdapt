from thetis import *

import numpy as np
from time import clock

from utils.adaptivity import *
from utils.callbacks import AdvectionCallback
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
    options.use_grad_div_viscosity_term = True              # Symmetric viscous stress
    options.use_lax_friedrichs_velocity = False             # TODO: This is a temporary fix
    options.simulation_export_time = op.dt * op.ndump
    options.simulation_end_time = op.Tend - 0.5 * op.dt
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = op.di()
    if not op.plotPVD:
        options.no_exports = True
    options.horizontal_velocity_scale = op.u_mag
    options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
    options.solve_tracer = True
    options.tracer_only = True  # Need use tracer-only branch to use this functionality
    options.horizontal_diffusivity = diffusivity
    options.use_lax_friedrichs_tracer = False
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
    if op.plotMetric:
        mFile = File(op.di() + "Metric2d.pvd")

    # Initialise domain and physical parameters
    V = op.mixedSpace(mesh)
    uv_2d, elev_2d = Function(V).split()  # Needed to load data into
    elev_2d.interpolate(eta0)
    uv_2d.interpolate(u0)
    tracer_2d = Function(FunctionSpace(mesh, "CG", 1))

    # Initialise parameters and counters
    nEle = mesh.num_cells()
    op.nVerT = mesh.num_vertices() * op.rescaling   # Target #Vertices
    mM = [nEle, nEle]  # Min/max #Elements
    Sn = nEle
    cnt = 0
    t = 0.

    adaptSolveTimer = 0.
    quantities = {}
    while cnt < op.cntT():
        adaptTimer = clock()
        P1 = FunctionSpace(mesh, "CG", 1)

        tracer = Function(P1).interpolate(tracer_2d)
        for l in range(op.nAdapt):                  # TODO: Test this functionality

            # Construct metric
            if cnt != 0:   # Can't adapt to zero concentration
                M = steadyMetric(tracer, op=op)

            # Adapt mesh and interpolate variables
            if cnt != 0:
                mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            if l < op.nAdapt-1:
                tracer = interp(mesh, tracer)

        if cnt != 0:
            if op.plotMetric:
                if op.nAdapt == 0:
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
        adapOpt.use_grad_div_viscosity_term = True                  # Symmetric viscous stress
        adapOpt.use_lax_friedrichs_velocity = False                 # TODO: This is a temporary fix
        adapOpt.simulation_export_time = op.dt * op.ndump
        adapOpt.simulation_end_time = t + op.dt * (op.rm - 0.5)
        adapOpt.timestepper_type = op.timestepper
        adapOpt.timestep = op.dt
        adapOpt.output_directory = op.di()
        if not op.plotPVD:
            adapOpt.no_exports = True
        adapOpt.horizontal_velocity_scale = op.u_mag
        adapOpt.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
        adapOpt.solve_tracer = True
        adapOpt.tracer_only = True  # Need use tracer-only branch to use this functionality
        adapOpt.horizontal_diffusivity = diffusivity
        adapOpt.use_lax_friedrichs_tracer = False
        adapOpt.tracer_source_2d = source
        adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d, tracer=tracer_2d)
        adapSolver.i_export = int(cnt / op.ndump)
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
        cnt += op.rm
        t += op.dt * op.rm
        av = op.printToScreen(int(cnt/op.rm+1), adaptTimer, solverTimer, nEle, Sn, mM, cnt * op.dt)
        adaptSolveTimer += adaptTimer + solverTimer

        # Extract fields for next step
        uv_2d, elev_2d = adapSolver.fields.solution_2d.split()
        tracer_2d = adapSolver.fields.tracer_2d

    # Output mesh statistics and solver times
    quantities['meanElements'] = av
    quantities['solverTimer'] = adaptSolveTimer
    quantities['adaptSolveTimer'] = adaptSolveTimer

    return quantities


def advect(mesh, u0, eta0, b, BCs={}, source=None, diffusivity=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    solvers = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased}

    return solvers[op.approach](mesh, u0, eta0, b, BCs, source, diffusivity, regen=regen, op=op)
