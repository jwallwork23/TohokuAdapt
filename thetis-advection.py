# Pure tracer advection using Thetis
# ==============================================

from scipy.interpolate import interp1d
from thetis import *
import math


outputdir = 'plots/channel2d_tracer'
mesh2d = PeriodicRectangleMesh(20, 100, 10., 50., direction='x')
print_output('Number of elements '+str(mesh2d.num_cells()))
print_output('Number of nodes '+str(mesh2d.num_vertices()))

# total duration in seconds
t_end = 500.
# estimate of max advective velocity used to estimate time step
u_mag = Constant(6.0)   # TODO
# export interval in seconds
t_export = 100.

# bathymetry
P1_2d = FunctionSpace(mesh2d, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')

# depth_oce = 20.0
# depth_riv = 5.0  # 5.0 closed
# bath_x = np.array([0, 100e3])
# bath_v = np.array([depth_oce, depth_riv])
#
#
# def bath(x, y, z):
#     padval = 1e20
#     x0 = np.hstack(([-padval], bath_x, [padval]))
#     vals0 = np.hstack(([bath_v[0]], bath_v, [bath_v[-1]]))
#     return interp1d(x0, vals0)(x)
#
#
# x_func = Function(P1_2d).interpolate(Expression('x[0]'))
# bathymetry_2d.dat.data[:] = bath(x_func.dat.data, 0, 0)
bathymetry_2d.assign(1.)


# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
options = solver_obj.options
options.simulation_export_time = t_export
options.simulation_end_time = t_end
options.output_directory = outputdir
options.horizontal_velocity_scale = u_mag
options.check_volume_conservation_2d = True
options.fields_to_export = ['uv_2d', 'elev_2d', 'tracer_2d']
options.solve_tracer = True
options.use_lax_friedrichs_tracer = False
# options.tracer_source_2d = q_source
options.timestepper_type = 'CrankNicolson'
options.timestep = 0.1
# initial conditions, piecewise linear function
elev_x = np.array([0, 30e3, 100e3])
elev_v = np.array([6, 0, 0])


# def elevation(x, y, z, x_array, val_array):
#     padval = 1e20
#     x0 = np.hstack(([-padval], x_array, [padval]))
#     vals0 = np.hstack(([val_array[0]], val_array, [val_array[-1]]))
#     return interp1d(x0, vals0)(x)
#
#
# x_func = Function(P1_2d).interpolate(Expression('x[0]'))
elev_init = Function(P1_2d)
# elev_init.dat.data[:] = elevation(x_func.dat.data, 0, 0, elev_x, elev_v)

uv_init = Function(VectorFunctionSpace(mesh2d, "CG", 1))
uv_init.assign(1.)

# Tracer initial field
x, y = SpatialCoordinate(mesh2d)
bell_r0 = 1000; bell_x0 = 30000; bell_y0 = 1500
bell = conditional(ge(0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0))),0.),
                   0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0))), 0. )
q_init = Function(P1_2d).interpolate(0.0 + bell)
# q_init = Function(P1_2d).assign(1.0)

solver_obj.assign_initial_conditions(elev=elev_init, tracer= q_init)

solver_obj.iterate()
