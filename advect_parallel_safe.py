from thetis import *
from firedrake.petsc import PETSc

import argparse
from time import clock
import datetime

from utils.callbacks import AdvectionCallback
from utils.setup import problem_domain
from utils.options import AdvectionOptions


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
    options.use_lax_friedrichs_tracer = False                   # TODO: This is a temporary fix
    options.tracer_family = op.tracer_family
    if op.tracer_family == 'cg':
        options.use_limiter_for_tracers = False
    options.tracer_source_2d = source
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = AdvectionCallback(solver_obj, parameters=op)
    solver_obj.add_callback(cb1, 'timestep')
    solver_obj.bnd_functions = BCs

    # Solve and extract timeseries / functionals
    quantities = {}
    solver_timer = clock()
    solver_obj.iterate()
    solver_timer = clock() - solver_timer
    quantities['J_h'] = cb1.get_val()          # Evaluate objective functional

    # Output mesh statistics and solver times
    quantities['mean_elements'] = mesh.num_cells()
    quantities['solver_timer'] = solver_timer

    return quantities


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="Finite element family, from {'dg', 'cg'}")
parser.add_argument("-o", help="Output data")
parser.add_argument("-level", help="Single mesh resolution")
parser.add_argument("-snes_view", help="Use PETSc snes view.")
args = parser.parse_args()
approach = 'FixedMesh'

# Set parameters
op = AdvectionOptions(approach=approach)
op.plot_pvd = bool(args.o)
op.tracer_family = args.f if args.f is not None else 'cg'
if bool(args.snes_view):
    op.solver_parameters['snes_view'] = True
level = int(args.level)

# Get data
mesh, u0, eta0, b, BCs, source, diffusivity = problem_domain(level, op=op)
quantities = FixedMesh(mesh, u0, eta0, b, BCs=BCs, source=source, diffusivity=diffusivity, op=op)
PETSc.Sys.Print("Mode: %s Approach: %s. Run: %d" % ('advection-diffusion', approach, level), comm=COMM_WORLD)
PETSc.Sys.Print("Run %d: Mean element count: %6d Timing %.1fs Objective: %.4e" % (level, quantities['mean_elements'], quantities['solver_timer'], quantities['J_h']), comm=COMM_WORLD)

# Save to disk
filename = 'outdata/AdvectionDiffusion/' + approach + '_' + date
errorFile = open(filename + '.txt', 'a+')   # Initialise file
errorFile.write('run %d: %d, %.1f, %.4e\n' % (level, quantities['mean_elements'], quantities['solver_timer'], quantities['J_h']))
errorFile.close()
