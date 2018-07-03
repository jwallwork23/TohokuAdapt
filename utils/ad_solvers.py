from thetis import *

import numpy as np
from time import clock

from utils.adaptivity import *
from utils.callbacks import AdvectionCallback
from utils.interpolation import interp, mixedPairInterp
from utils.setup import problemDomain


__all__ = ["advect"]


default_BC = {'shallow_water': {}, 'tracer': {}}


def fixedMesh(mesh, u0, eta0, b, BCs=default_BC, source=None, diffusivity=None, **kwargs):
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
    if not op.plotpvd:
        options.no_exports = True
    options.horizontal_velocity_scale = op.u_mag
    options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
    options.solve_tracer = True
    options.tracer_only = True  # Need use tracer-only branch to use this functionality
    options.horizontal_diffusivity = Constant(diffusivity)
    options.use_lax_friedrichs_tracer = False
    if source is not None:
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


def advect(mesh, u0, eta0, b, BCs=default_BC, source=None, **kwargs):
    op = kwargs.get('op')
    regen = kwargs.get('regen')
    solvers = {'fixedMesh': fixedMesh}

    return solvers[op.approach](mesh, u0, eta0, b, BCs, source, regen=regen, op=op)
