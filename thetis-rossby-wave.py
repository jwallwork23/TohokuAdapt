from thetis import *

import numpy as np

import utils.forms as form
import utils.options as opt


# Get parameters
op = opt.Options(Tend=120.,
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

# Set initial/boundary conditions and bathymetry
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = form.analyticHuang(V)
uv0, elev0 = q_.split()
BCs = {}
BCs[1] = {'uv': Function(V.sub(0))}     # Zero velocity on South boundary
BCs[2] = {'uv': Function(V.sub(0))}     # Zero velocity on North boundary
b = Function(FunctionSpace(mesh, "CG", 1)).assign(1.)
physical_constants['g_grav'] = Constant(1.)

# Construct solver
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = True
options.use_grad_depth_viscosity_term = False
options.coriolis_frequency = y
options.timestepper_type = op.timestepper
solver_obj.create_equations()
options.timestep = min(np.abs(solver_obj.compute_time_step().dat.data))
options.simulation_export_time = options.timestep * op.ndump
options.simulation_end_time = op.Tend
options.output_directory = 'plots/rossby-wave'
solver_obj.assign_initial_conditions(elev=elev0, uv=uv0)
solver_obj.bnd_functions['shallow_water'] = BCs
solver_obj.iterate()
