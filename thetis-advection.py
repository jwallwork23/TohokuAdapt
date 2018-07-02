# Pure tracer advection using Thetis
# ==============================================
#
# Emulates TELEMAC-2D 'point discharge with diffusion 23-tradif' validation
# experiment conducted by D. Lebris, July 2015.
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
import math


outputdir = 'plots/channel2d_tracer'
mesh2d = RectangleMesh(100, 20, 50., 10.)
print_output('Number of elements '+str(mesh2d.num_cells()))
print_output('Number of nodes '+str(mesh2d.num_vertices()))

t_end = 50.             # Total duration in seconds
u_mag = Constant(1.0)   # (Estimate of) max advective velocity used to estimate time step
t_export = 1.           # Export interval in seconds

# Bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')
bathymetry_2d.assign(1.)

# Tracer initial field
x, y = SpatialCoordinate(mesh2d)
bell_r0 = 0.457; bell_x0 = 1.; bell_y0 = 5.
bell = conditional(ge(0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0))),0.),
                   0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0))), 0. )
q_source = Function(P1_2d).interpolate(0.0 + bell)

# Objective functional parameters
x0 = 25.
y0 = 7.5
r = 0.5

# --- Create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = u_mag
# options.check_tracer_conservation = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
options.solve_tracer = True
options.tracer_only = True      # Need use tracer-only branch to use this functionality
# options.horizontal_diffusivity = Constant(0.1)
options.use_lax_friedrichs_tracer = False
options.tracer_source_2d = q_source
options.timestepper_type = 'CrankNicolson'
options.timestep = 0.1
# options.use_nonlinear_equations = False

# Initial conditions
elev_init = Function(P1_2d)
uv_init = Function(VectorFunctionSpace(mesh2d, "CG", 1))
uv_init.interpolate(Expression([1., 0.]))

# Boundary conditions
solver_obj.bnd_functions = {'shallow_water': {}, 'tracer': {}}
solver_obj.bnd_functions['shallow_water']['uv']= {1: uv_init}
solver_obj.bnd_functions['shallow_water']['uv'][2] = uv_init
# solver_obj.bnd_functions['tracer'] = {1: Function(P1_2d)}

# Solve
solver_obj.assign_initial_conditions(elev=elev_init, uv=uv_init)
assert(options.timestep < min(solver_obj.compute_time_step().dat.data))     # Check CFL condition
solver_obj.iterate()
