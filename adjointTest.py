from firedrake import *
from firedrake_adjoint import *

from time import clock

import utils.forms as form
import utils.mesh as msh
import utils.options as opt

op = opt.Options()
dt_meas = dt        # Time measure
dt = 3.5            # Time step
Dt = Constant(dt)
nEle = op.meshes[1]
T = 1500.           # End time
t = 0.
cnt = 0

# Define Mesh, FunctionSpace and apply ICs
mesh, eta0, b = msh.TohokuDomain(nEle)
x, y = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
q_ = Function(V)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]), annotate=False)
eta_.interpolate(eta0, annotate=False)
q = Function(V)
q.assign(q_)
u, eta = q.split()
u.rename("uv_2d")
eta.rename("elev_2d")

# Create adjoint variables
dual = Function(V)
dual_u, dual_e = dual.split()
dual_u.rename("Adjoint velocity")
dual_e.rename("Adjoint elevation")

# Define variational problem: advection-diffusion with no-normal-flow BCs
qt = TestFunction(V)
forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt, allowNormalFlow=False), q)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

# Establish objective functional associated with the space-time integral of concentration in a certain region
J = form.objectiveFunctionalSW(q)

# Solve primal problem
primalTimer = clock()
while t < T:
    forwardSolver.solve()
    q_.assign(q)
    if t == 0.:
        adj_start_timestep()
    elif t >= T:
        adj_inc_timestep(time=t, finished=True)
    else:
        adj_inc_timestep(time=t, finished=False)
    t += dt
    cnt += 1
cnt -= 1
primalTimer = clock()-primalTimer
print('Forward run time: %.3fs' % primalTimer)

# Visualise each run
parameters["adjoint"]["stop_annotating"] = True                 # Stop registering equations
# adj_html("outdata/visualisations/forward.html", "forward")
# adj_html("outdata/visualisations/adjoint.html", "adjoint")

# Solve dual problem
store = True
dualTimer = clock()
for (variable, solution) in compute_adjoint(J):
    if store:
        dual.assign(variable, annotate=False)
        cnt -= 1
        store = False
    else:
        store = True
    if cnt == 0:
        break
dualTimer = clock()-dualTimer
print('Adjoint run time:   %.3fs' % dualTimer)
assert(cnt == 0)
print('Adjoint run %.3fx slower than forward run.' % (dualTimer/primalTimer))
