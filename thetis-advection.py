# Tracer advection and diffusiion in a constant velocity field using Thetis
# ============================================================================
#
# Imposes initial and boundary conditions for the shallow water equations such
# that the free surface displacement is zero and fluid velocity is a constant
# 1 m/s. Tracer concentration is initially zero and has a source equivalent to
# 1 m^3/s (in the 3D case).
#
# Code structure is based on `channel2d_tracer` by A. Angeloudis.
#
# Currently requires `tracer-only` branch of Thetis: https://github.com/thetisproject/thetis/tree/tracer-only


from thetis import *

from utils.callbacks import AdvectionCallback
from utils.options import AdvectionOptions
from utils.setup import problem_domain


op = AdvectionOptions()
op.timesteps_per_export = 100

# Parameter choice 1: pure advection
level = 2
op.bell_r0 = 2.
op.bell_x0 = 5.
op.bell_y0 = 5.
op.end_time = 40.
op.timestep = 0.01
op.diffusivity = 0.
source_off = True

# # Parameter choice 2: advection and diffusion
# level = 2
# op.bell_r0 = 2.
# op.bell_x0 = 5.
# op.bell_y0 = 5.
# op.end_time = 40.
# op.timestep = 0.01
# op.diffusivity = 1e-3
# source_off = True

# # Parameter choice 3: advection and diffusion with a constant source
# level = 2
# op.bell_r0 = 2.
# op.bell_x0 = 5.
# op.bell_y0 = 5.
# op.end_time = 40.
# op.timestep = 0.01
# op.diffusivity = 1e-3

# # Parameter choice 4 (TELEMAC-2D point discharge without diffusion)
# level = 1
# op.bell_r0 = 0.457
# op.bell_x0 = 1.5
# op.bell_y0 = 5.
# op.end_time = 50.
# op.timestep = 0.1
# op.diffusivity = 0.

# # Parameter choice 5 (TELEMAC-2D point discharge with diffusion)
# level = 1
# op.bell_r0 = 0.457
# op.bell_x0 = 1.5
# op.bell_y0 = 5.
# op.end_time = 50.
# op.timestep = 0.1
# op.diffusivity = 0.1

# Setup domain
mesh, u0, eta0, b, BCs, source, diffusivity = problem_domain(level, op=op)
print_output('Number of elements %d' % mesh.num_cells())
print_output('Number of nodes %d' % mesh.num_vertices())
print_output("Souce volume = %.4f" % assemble(source * dx))
u_mag = Constant(1.0)   # (Estimate of) max advective velocity used to estimate time step

# --- Create solver ---
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.simulation_export_time = op.timesteps_per_export * op.timestep
options.simulation_end_time = op.end_time
options.output_directory = op.directory()
options.horizontal_velocity_scale = u_mag
options.fields_to_export = ['tracer_2d']
options.solve_tracer = True
options.tracer_only = True              # Need use tracer-only branch to use this functionality
options.horizontal_diffusivity = diffusivity
options.use_lax_friedrichs_tracer = False
options.tracer_source_2d = None if source_off else source
options.timestepper_type = 'CrankNicolson'
options.timestep = op.timestep

solver_obj.assign_initial_conditions(elev=eta0, uv=u0, tracer=source if source_off else None)
try:
    cdt = min(solver_obj.compute_time_step().dat.data)
    assert(options.timestep < cdt)     # Check CFL condition
except:
    raise ValueError("Chosen timestep %.2fs is smaller than recommended timestep %.2fs" % (options.timestep, cdt))

# Establish callbacks and solve
cb1 = AdvectionCallback(solver_obj)
cb1.op = op
solver_obj.add_callback(cb1, 'timestep')
solver_obj.iterate()
print_output('Objective value = %.4f' % cb1.get_val())
