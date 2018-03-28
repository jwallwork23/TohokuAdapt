from thetis import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock   # Need use Sebastian's `linear-solver` branch of pyadjoint

import utils.error as err


# Establish Mesh, initial condition and bathymetry
mesh = SquareMesh(16, 16, 2 * pi, 2 * pi)
x, y = SpatialCoordinate(mesh)
P1_2d = FunctionSpace(mesh, "CG", 1)
eta0 = Function(P1_2d).interpolate(1e-3 * exp(-(pow(x - pi, 2) + pow(y - pi, 2))))
b = Function(P1_2d).assign(0.1)

# Establish adjoint variables and indicator function
V = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "DG", 1)
dual = Function(V)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Get solver parameter values and construct solver
dt = 0.025
ndump = 10
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = 'dg-dg'
options.use_nonlinear_equations = False
options.use_grad_depth_viscosity_term = False
options.use_grad_div_viscosity_term = False
options.simulation_export_time = dt * ndump
options.simulation_end_time = 2.5
options.timestepper_type = 'CrankNicolson'
options.timestep = dt
options.output_directory = 'plots/pyadjointTest/'
options.export_diagnostics = False
solver_obj.assign_initial_conditions(elev=eta0)
cb = err.ObjectiveSWCallback(solver_obj)        # Extract objective functional at each timestep for use in pyadjoint
solver_obj.add_callback(cb, 'timestep')
solver_obj.iterate()

# Assemble objective functional
Jfuncs = cb.__call__()[1]
J = 0
for i in range(1, len(Jfuncs)):
    J += 0.5*(Jfuncs[i-1] + Jfuncs[i])*dt

# Compute gradient
dJdb = compute_gradient(J, Control(b))
File('plots/pyadjointTest/gradient.pvd').write(dJdb)
print("Norm of gradient = %e" % dJdb.dat.norm)

# Extract adjoint solutions
tape = get_working_tape()
# tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
adjointFile = File('plots/pyadjointTest/adjoint.pvd')
for i in range(len(solve_blocks)-1, -1, -1):
    try:
        dual.assign(solve_blocks[i].adj_sol)
        dual_u, dual_e = dual.split()
    except:
        print('Block appears to have Nonetype',)
    if i % ndump == 0:
        adjointFile.write(dual_u, dual_e, time=dt*i)
        print('t = %.2fs' % (dt*i))
