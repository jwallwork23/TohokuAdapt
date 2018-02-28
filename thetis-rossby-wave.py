from thetis import *

import utils.forms as form
import utils.options as opt


# Set solver parameters
op = opt.Options(Tend=120.,
                 family='dg-dg',
                 ndump=12,
                 dt=0.1)        # In Matt's parameters timestep=0.1
dirName = 'plots/rossby-wave/'

# Define Mesh
n = 4
lx = 48
ly = 24
mesh = PeriodicRectangleMesh(lx*n, ly*n, lx, ly, direction="x")
xy = Function(mesh.coordinates)
xy.dat.data[:, :] -= [lx/2, ly/2]
mesh.coordinates.assign(xy)
x, y = SpatialCoordinate(mesh)

# Define FunctionSpaces
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
P1 = FunctionSpace(mesh, "CG", 1)

# Define physical parameters
physical_constants['g_grav'] = Constant(1.)     # Gravitational acceleration
b = Function(P1).assign(1.)
f = Function(P1).interpolate(y)

# # Plot analytic solution
# t = 0.
# solFile = File(dirName + 'analytic.pvd')
# print('Generating analytic solution')
# while t < op.Tend:
#     q = form.solutionHuang(V, t=t)
#     u, eta = q.split()
#     u.rename('Analytic fluid velocity')
#     eta.rename('Analytic free surface')
#     solFile.write(u, eta, time=t)
#     print('t = %.1fs' % t)
#     t += op.ndump * dt

# Assign initial and boundary conditions
q0 = form.solutionHuang(V, t=0.)
uv0, elev0 = q0.split()
BCs = {1: {'uv': Constant(0.)}, 2: {'uv': Constant(0.)}}    # Zero velocity on South & North boundaries

# Construct solver
print('Generating numerical solution')
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = True
options.use_grad_depth_viscosity_term = True       # In Matt's parameters viscosity=1e-6
options.coriolis_frequency = f
options.simulation_export_time = op.dt * op.ndump
options.simulation_end_time = op.Tend
options.timestepper_type = op.timestepper
# solver_obj.create_equations()
# options.timestep = min(np.abs(solver_obj.compute_time_step().dat.data))
options.timestep = op.dt
options.output_directory = dirName
solver_obj.assign_initial_conditions(elev=elev0, uv=uv0)
solver_obj.bnd_functions['shallow_water'] = BCs
solver_obj.iterate()

# Sanity check
print('gravity', float(physical_constants['g_grav'].dat.data))
bval = min(b.dat.data)
assert(bval == max(b.dat.data))
print('bathy', bval)
File(dirName+'Coriolis.pvd').write(options.coriolis_frequency)
