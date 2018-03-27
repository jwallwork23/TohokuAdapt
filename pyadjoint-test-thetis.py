from thetis import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock   # Need use Sebastian's `linear-solver` branch of pyadjoint
from firedrake import Expression                # Annotation of Expressions not currently supported in firedrake_adjoint

import numpy as np

import utils.error as err       # Contains callbacks for objective functional computation and recording to tape

Ts = 0.5
T = 2.5
dt = 0.025
ndump = 10
di = 'plots/pyadjointTest/'

# Set up Mesh
lx = 2 * np.pi
n = pow(2, 4)
mesh_H = SquareMesh(n, n, lx, lx)
x, y = SpatialCoordinate(mesh_H)
P1_2d = FunctionSpace(mesh_H, "CG", 1)
eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
b = Function(P1_2d).assign(0.1)

# Define initial FunctionSpace and variables of problem and apply initial conditions
V_H = VectorFunctionSpace(mesh_H, "DG", 1) * FunctionSpace(mesh_H, "DG", 1)
q = Function(V_H)
uv_2d, elev_2d = q.split()
elev_2d.interpolate(eta0)
uv_2d.rename("uv_2d")
elev_2d.rename("elev_2d")

# Set up adjoint variables
dual = Function(V_H)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Define indicator function
k = Function(V_H)
k0, k1 = k.split()
k1.interpolate(Expression('(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (0., pi/2, pi/2, 3*pi/2)))
# J = assemble(inner(k, q) * dx)

# Get solver parameter values and construct solver
solver_obj = solver2d.FlowSolver2d(mesh_H, b)
solver_obj.create_equations()
options = solver_obj.options
options.element_family = 'dg-dg'
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.use_grad_div_viscosity_term = False
options.simulation_export_time = dt * ndump
options.simulation_end_time = T
options.timestepper_type = 'CrankNicolson'
options.timestep = dt
options.output_directory = di
options.export_diagnostics = False

# Output OF values
cb1 = err.ShallowWaterCallback(solver_obj)
cb1.append_to_log = False
cb1.export_to_hdf5 = False
solver_obj.add_callback(cb1, 'timestep')

# Get OF values
cb2 = err.ObjectiveSWCallback(solver_obj)   # TODO: Callback for adding OF in pyadjoint sense?
cb2.append_to_log = False
cb2.export_to_hdf5 = False
solver_obj.add_callback(cb2, 'timestep')

# Apply ICs and time integrate
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.iterate()

# Get objective functional and its value on the current forward solution
print("Objective value = %.4e" % (cb1.__call__()[1]))
Jfuncs = cb2.__call__()[1]
J = 0
for i in range(1, len(Jfuncs)):
    J += 0.5*(Jfuncs[i-1] + Jfuncs[i])*dt
print(type(J))    # Should be a functional, not a number

# Extract adjoint variables
t = T
dJdb = compute_gradient(J, Control(b)) # Need compute gradient or tlm in order to extract adjoint solutions
tape = get_working_tape()
tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
adjointFile = File(di + 'adjoint.pvd')
for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    dual_u, dual_e = dual.split()
    if t % ndump:
        print('t = %.2fs' % t)
    adjointFile.write(dual_u, dual_e, time=t)
    t -= dt
