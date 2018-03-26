from thetis import *

import utils.forms as form
import utils.misc as msc


# Set solver parameters
periodic = False
dt = 0.1            # In Matt's parameters dt=0.1
T = 120.
ndump = 12
dirName = 'plots/rossby-wave/'

# Define Mesh
n = 1
lx = 48
ly = 24
mesh = PeriodicRectangleMesh(lx*n, ly*n, lx, ly, direction="x") if periodic else RectangleMesh(3*lx*n, ly*n, 3*lx, ly)
xy = Function(mesh.coordinates)
xy.dat.data[:, :] -= [lx/2, ly/2] if periodic else [3*lx/2, ly/2]
mesh.coordinates.assign(xy)

# Define FunctionSpaces and  physical fields
V = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1)
P1 = FunctionSpace(mesh, "CG", 1)
try:
    assert (float(physical_constants['g_grav'].dat.data) == 1.)
except:
    physical_constants['g_grav'].assign(1.)
b = Function(P1).assign(1.)
f = Function(P1).interpolate(SpatialCoordinate(mesh)[1])

# Assign initial and boundary conditions
q = form.solutionHuang(V, t=0.)
uv_2d, elev_2d = q.split()
BCs = {1: {'uv': Constant(0.)}, 2: {'uv': Constant(0.)}}        # No-slip BCs on South & North boundaries
if not periodic:
    BCs[3] = {'uv': Constant(0.)}
    BCs[4] = {'uv': Constant(0.)}

# Construct solver
print('Generating numerical solution')
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = 'dg-dg'
options.use_nonlinear_equations = True
options.use_grad_depth_viscosity_term = False                   # In Matt's parameters viscosity=1e-6
options.use_grad_div_viscosity_term = False
options.coriolis_frequency = f
File(dirName+'Coriolis.pvd').write(options.coriolis_frequency)  # Plot Coriolis frequency
options.simulation_export_time = dt * ndump
options.simulation_end_time = T
options.timestepper_type = 'CrankNicolson'
# solver_obj.create_equations()
# options.timestep = min(np.abs(solver_obj.compute_time_step().dat.data))
options.timestep = dt
options.output_directory = dirName
solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
solver_obj.bnd_functions['shallow_water'] = BCs
solver_obj.iterate()

# # Check residual approximation works using Thetis forms
# indexStr = msc.indexString(int(T/(dt * ndump)))
# with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
#     loadVel.load(uv_2d)
#     loadVel.close()
# with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
#     loadElev.load(elev_2d)
#     loadElev.close()
# V_oi = VectorFunctionSpace(mesh, "DG", 2) * FunctionSpace(mesh, "DG", 2)
# q_oi = Function(V)
# uv_oi, elev_oi = q_oi.split()
# uv_oi.interpolate(uv_2d)
# elev_oi.interpolate(elev_2d)
# eqns = ShallowWaterEquations(V_oi, b)
# res = eqns.residual(label, q_oi, q_oi_, fields, fields_old, bnd_conditions)     # TODO: this won't work yet
