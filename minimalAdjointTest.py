from firedrake import *
from firedrake_adjoint import *
from time import clock

dt_meas = dt        # Time measure
dt = 0.04           # Time step
n = 2**8
lx = 4
ly = 1
T = 2.4             # End time
t = 0.
cnt = 0
primalFile = File('plots/minTest/primal.pvd')
dualFile = File('plots/minTest/dual.pvd')

# Define Mesh, FunctionSpace and apply IC
mesh = RectangleMesh(n * lx, n * ly, 4, 1)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
c_ = Function(V, name='Prev').interpolate(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), annotate=False)
c = Function(V, name='Concentration')
dual = Function(V, name='Adjoint')

# Define variational problem: advection-diffusion with no-normal-flow BCs
ct = TestFunction(V)
cm = 0.5 * (c + c_) # Crank-Nicolson timestepping
w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]), annotate=False)
F = ((c - c_) * ct / Constant(dt) + inner(grad(cm), w * ct) + Constant(1e-3) * inner(grad(cm), grad(ct))) * dx

# Establish objective functional associated with the space-time integral of concentration in a certain region
expr = '(x[0] > 2.75) & (x[0] < 3.25) & (x[1] > 0.25) & (x[1] < 0.75) ? 1. : 0.'
indicator = Function(V).interpolate(Expression(expr), annotate=False)
J = Functional(c * indicator * dx * dt_meas)

# Solve primal problem
primalTimer = clock()
while t < T:
    solve(F == 0, c)
    c_.assign(c)
    primalFile.write(c)
    if t == 0.:
        adj_start_timestep()
    elif t >= T:
        adj_inc_timestep(time=t, finished=True)
    else:
        adj_inc_timestep(time=t, finished=False)
    t += dt
    cnt += 1
cnt -= 1
print('Primal run time: %.3fs' % (clock()-primalTimer))

# Visualise each run
parameters["adjoint"]["stop_annotating"] = True                 # Stop registering equations
adj_html("outdata/visualisations/forward.html", "forward")
adj_html("outdata/visualisations/adjoint.html", "adjoint")

# Solve dual problem
store = True
dualTimer = clock()
for (variable, solution) in compute_adjoint(J):
    if store:
        dual.assign(variable, annotate=False)
        dualFile.write(dual)
        cnt -= 1
        store = False
    else:
        store = True
    if cnt == 0:
        break
print('Dual run time:   %.3fs' % (clock()-dualTimer))
assert(cnt == 0)
