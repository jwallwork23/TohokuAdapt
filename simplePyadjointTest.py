from firedrake import *
from firedrake_adjoint import *
from fenics_adjoint.solving import SolveBlock

import utils.forms as form


forwardFile = File('plots/pyadjoint_test/forward.pvd')
adjointFile = File('plots/pyadjoint_test/adjoint.pvd')

# Load Mesh(es)
n = 8
mesh_H = SquareMesh(4 * n, n, 4, 1)
x, y = SpatialCoordinate(mesh_H)
P1 = FunctionSpace(mesh_H, "CG", 1)
w = Constant([1, 0])
c_ = Function(P1).interpolate(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04))
c = TrialFunction(P1)

# Specify physical and solver parameters
dt = 0.05
Dt = Constant(dt)
Tstart = 0.
Tend = 2.4
nu = Constant(1e-3)

# Define adjoint variable and indicator function
dual = Function(P1)
dual.rename("Adjoint")
def indicator(P1):
    k = Function(P1, name='Region of interest', annotate=False)
    k.interpolate(Expression('(x[0] > 2.5) & (x[0] < 3.5) & (x[1] > 0.1) & (x[1] < 0.9) ? 1. : 0.'), annotate=False)
    return k
k = form.indicator(P1)

# Define variational problem
ct = TestFunction(P1)
a = (c * ct / Dt) * dx
L = (c_ * ct / Dt - inner(grad(c_), w * ct) - nu * inner(grad(c_), grad(ct))) * dx
c = Function(P1)
prob = LinearVariationalProblem(a, L, c)
solv = LinearVariationalSolver(prob)

t = 0.
forwardFile.write(c_, time=t)
# Jfuncs = [assemble(inner(k, c_) * dx)]
J = assemble(k * c_ * dx)
while t < Tend + dt:
    solv.solve()
    c_.assign(c)
    # Jfuncs.append(assemble(inner(k, c_) * dx))  # Update OF
    forwardFile.write(c_, time=t)
    print('t = %.2fs' % t)
    t += dt
t -= dt

# # Establish OF
# J = 0
# for i in range(1, len(Jfuncs)):
#     J += 0.5*(Jfuncs[i-1] + Jfuncs[i])*dt

dJdnu = compute_gradient(J, Control(nu))
tape = get_working_tape()
tape.visualise()
solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]

for i in range(len(solve_blocks)-1, -1, -1):
    dual.assign(solve_blocks[i].adj_sol)
    adjointFile.write(dual, time=t)
    t -= dt

File('plots/pyadjoint_test/gradient.pvd').write(dJdnu)
