from thetis import *

import numpy as np

import utils.forms as form
import utils.options as opt


# Get parameters
op = opt.Options(Tstart=0.,
                 Tend=120.,
                 family='dg-dg',
                 ndump=6)

# Define domain
n = 4
lx = 48
ly = 24
mesh = PeriodicRectangleMesh(lx * n, ly * n, lx, ly, direction="x")
xy = Function(mesh.coordinates)
xy.dat.data[:, 0] -= 24.
xy.dat.data[:, 1] -= 12.
mesh.coordinates.assign(xy)
x, y = SpatialCoordinate(mesh)

# Set initial condition and bathymetry
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = form.analyticHuang(V)
uv0, elev0 = q_.split()
b = Function(FunctionSpace(mesh, "CG", 1)).assign(1.)

# Set Coriolis frequency and gravitational acceleration
f0 = 0.     # TODO: put in options
beta = 1.   # TODO: put in options
f = f0 + beta * y
physical_constants['g_grav'] = Constant(1.)

# Get timestep
solver_obj = solver2d.FlowSolver2d(mesh, b)
solver_obj.create_equations()
dt = min(np.abs(solver_obj.compute_time_step().dat.data))

# Solve
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.coriolis_frequency = f
options.simulation_export_time = dt * op.ndump
options.simulation_end_time = op.Tend
options.timestepper_type = op.timestepper
options.timestep = dt
options.output_directory = 'plots/rossby-wave'
solver_obj.assign_initial_conditions(elev=elev0, uv=uv0)
solver_obj.iterate()
