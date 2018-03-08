from firedrake import *

import utils.forms as form
import utils.options as opt


# Set solver parameters and plot directories
periodic = True
op = opt.Options(family='dg-cg',
                 # timestepper='ImplicitEuler',
                 timestepper='CrankNicolson',
                 dt=0.1,
                 Tend=120.,
                 ndump=12,
                 g=1.)
Dt = Constant(op.dt)
dirName = "plots/rossby-wave/"
forwardFile = File(dirName + "forwardRW.pvd")
solFile = File(dirName + 'analytic.pvd')

# Define Mesh
n = 5           # Which gives dx ~ 0.283 > 0.1 = dt
lx = 48
ly = 24
mesh = PeriodicRectangleMesh(lx*n, ly*n, lx, ly, direction="x") if periodic else RectangleMesh(lx*n, ly*n, lx, ly)
xy = Function(mesh.coordinates)
xy.dat.data[:, :] -= [lx/2, ly/2]
mesh.coordinates.assign(xy)

# Define FunctionSpaces and physical fields
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
P1 = FunctionSpace(mesh, "CG", 1)
b = Function(P1).assign(1.)
f = Function(P1).interpolate(SpatialCoordinate(mesh)[1])

# Assign initial and boundary conditions
q_ = form.solutionHuang(V, t=0.)
u_, eta_ = q_.split()
u_.rename('Velocity')
eta_.rename('Elevation')
bc = DirichletBC(V.sub(0), [0, 0], [1, 2] if periodic else 'on_boundary')

# Define variational problem
qt = TestFunction(V)
(w, xi) = (as_vector((qt[0], qt[1])), qt[2])
q = Function(V)
u_, eta_ = split(q_)
u, eta = split(q)
a1, a2 = form.timestepCoeffs(op.timestepper)
B = (inner(u, w) + eta * xi) / Dt * dx                    # LHS bilinear form
L = (inner(u_, w) + eta_ * xi) / Dt * dx                  # RHS linear functional
B += a1 * op.g * inner(grad(eta), w) * dx if op.space2 == "CG" else - (a1 * op.g * eta * div(w)) * dx
L -= a2 * op.g * inner(grad(eta_), w) * dx if op.space2 == "CG" else - (a2 * op.g * eta_ * div(w)) * dx
B -= a1 * inner(b * u, grad(xi)) * dx                      # No integration by parts
L += a2 * inner(b * u_, grad(xi)) * dx                     #           "
B += a1 * f * inner(as_vector((-u[1], u[0])), w) * dx     # Rotational terms
L -= a2 * f * inner(as_vector((-u_[1], u_[0])), w) * dx   #           "
F = B - L
forwardProblem = NonlinearVariationalProblem(F, q, bcs=bc)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=opt.Options().params)
u_, eta_ = q_.split()
u, eta = q.split()

# Initialise counters and solve numerically
t = 0.
cnt = 0
forwardFile.write(u_, eta_, time=t)
while t < op.Tend:
    forwardSolver.solve()
    q_.assign(q)
    if cnt % op.ndump == 0:
        forwardFile.write(u_, eta_, time=t)
        print('t = %.2fs' % t)
    t += op.dt
    cnt += 1

# # Plot analytic solution
# t = 0.
# print('Generating analytic solution')
# while t < op.Tend:
#     q = form.solutionHuang(V, t=t)
#     u, eta = q.split()
#     u.rename('Analytic fluid velocity')
#     eta.rename('Analytic free surface')
#     solFile.write(u, eta, time=t)
#     print('t = %.1fs' % t)
#     t += op.ndump * op.dt
