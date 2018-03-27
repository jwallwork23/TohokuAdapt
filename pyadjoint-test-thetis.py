from thetis import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock   # Need use Sebastian's `linear-solver` branch of pyadjoint
from firedrake import Expression                # Annotation of Expressions not currently supported in firedrake_adjoint

import utils.error as err       # Contains callbacks for objective functional computation and recording to tape

T = 2.5
dt = 0.025
ndump = 10
adjointFile = File('plots/pyadjointTest/adjoint.pvd')

# Set up Mesh, FunctionSpaces, etc
mesh = SquareMesh(16, 16, 2 * pi, 2 * pi)
x, y = SpatialCoordinate(mesh)
P1_2d = FunctionSpace(mesh, "CG", 1)
eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - pi, 2) + pow(y - pi, 2))))
b = Function(P1_2d).assign(0.1)
V = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1)

# Set up adjoint variables
dual = Function(V)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Define indicator function
k = Function(V)
k0, k1 = k.split()
k1.interpolate(Expression('(x[0] > %f) & (x[0] < %f) & (x[1] > %f) & (x[1] < %f) ? 1. : 0.' % (0., pi/2, pi/2, 3*pi/2)))

# Get solver parameter values and construct solver
solver_obj = solver2d.FlowSolver2d(mesh, b)
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
options.output_directory = 'plots/pyadjointTest/'
options.export_diagnostics = False
solver_obj.assign_initial_conditions(elev=eta0)

# Apply Callbacks and time integrate
cb1 = err.ShallowWaterCallback(solver_obj)  # Compute objective value on current solution
cb1.append_to_log = False
cb1.export_to_hdf5 = False
solver_obj.add_callback(cb1, 'timestep')
cb2 = err.ObjectiveSWCallback(solver_obj)   # Extract objective functional at each timestep for use in pyadjoint
cb2.append_to_log = False
cb2.export_to_hdf5 = False
solver_obj.add_callback(cb2, 'timestep')
solver_obj.iterate()

# Assemble objective functional and its value on the current forward solution
print("Objective value = %.4e" % (cb1.__call__()[1]))
Jfuncs = cb2.__call__()[1]
J = 0
for i in range(1, len(Jfuncs)):
    J += 0.5*(Jfuncs[i-1] + Jfuncs[i])*dt

# Extract adjoint variables
dJdb = compute_gradient(J, Control(b))  # Need compute gradient or tlm in order to extract adjoint solutions
tape = get_working_tape()
tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    dual_u, dual_e = dual.split()
    if i % ndump == 0:
        adjointFile.write(dual_u, dual_e, time=dt*i)
        print('t = %.2fs' % (dt*i))
