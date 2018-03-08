from firedrake import *

import utils.forms as form
import utils.options as opt


# Set solver parameters and plot directories
periodic = True
op = opt.Options(dt=0.1,
                 Tend=120.,
                 ndump=12,
                 g=1.)

Dt = Constant(op.dt)
dirName = "plots/rossby-wave/"
forwardFile = File(dirName + "forwardRW.pvd")
solFile = File(dirName + 'analytic.pvd')

# Define Mesh
n = 5
lx = 48
ly = 24
mesh = PeriodicRectangleMesh(lx*n, ly*n, lx, ly, direction="x") if periodic else RectangleMesh(lx*n, ly*n, lx, ly)
xy = Function(mesh.coordinates)
xy.dat.data[:, :] -= [lx/2, ly/2]
mesh.coordinates.assign(xy)
x, y = SpatialCoordinate(mesh)

# Define FunctionSpaces
V = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)

# Define physical fields
b = Function(P1).assign(1.)
f = Function(P1).interpolate(y)

# Assign initial and boundary conditions
q_ = form.solutionHuang(V, t=0.)
uv_, elev_ = q_.split()
uv_.rename('Velocity')
elev_.rename('Elevation')
bdyTag = [1, 2] if periodic else 'on_boundary'
bc = DirichletBC(V.sub(0), [0, 0], bdyTag)
q = Function(V)
uv, elev = q.split()
uv_, elev_ = split(q_)
uv, elev = split(q)

# Initialise counters
t = 0.
cnt = 0

# Define variational problem
qt = TestFunction(V)
(w, xi) = (as_vector((qt[0], qt[1])), qt[2])
a1, a2 = form.timestepCoeffs(op.timestepper)
B = (inner(uv, w) + elev * xi) / Dt * dx                    # LHS bilinear form
L = (inner(uv_, w) + elev_ * xi) / Dt * dx                  # RHS linear functional
if V.sub(1).ufl_element().family() == 'Lagrange':
    B += a1 * op.g * inner(grad(elev), w) * dx
    L -= a2 * op.g * inner(grad(elev_), w) * dx
else:
    B -= a1 * op.g * elev * div(w) * dx
    L += a2 * op.g * elev_ * div(w) * dx
B -= a1 * inner(b * uv, grad(xi)) * dx                      # No integration by parts
L += a2 * inner(b * uv_, grad(xi)) * dx                     #           "
B += a1 * f * inner(as_vector((-uv[1], uv[0])), w) * dx     # Rotational terms
L -= a2 * f * inner(as_vector((-uv_[1], uv_[0])), w) * dx   #           "
F = B - L
forwardProblem = NonlinearVariationalProblem(F, q, bcs=bc)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=opt.Options().params)
uv_, elev_ = q_.split()
uv, elev = q.split()

# Solve numerically
forwardFile.write(uv_, elev_, time=t)
while t < op.Tend:
    forwardSolver.solve()
    q_.assign(q)
    if cnt % op.ndump == 0:
        forwardFile.write(uv_, elev_, time=t)
        print('t = %.2fs' % t)
    t += op.dt
    cnt += 1

# Plot analytic solution
t = 0.
print('Generating analytic solution')
while t < op.Tend:
    q = form.solutionHuang(V, t=t)
    u, eta = q.split()
    u.rename('Analytic fluid velocity')
    eta.rename('Analytic free surface')
    solFile.write(u, eta, time=t)
    print('t = %.1fs' % t)
    t += op.ndump * op.dt
