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
BCs = DirichletBC(V.sub(0), [0, 0], [1, 2] if periodic else 'on_boundary')

# Define variational problem
q = Function(V)
u, eta = q.split()
F = form.weakResidualSW(q, q_, b, Dt, coriolisFreq=f, nonlinear=True, neumann=True, op=op)
# TODO: I think this extra BC needs removing.
forwardProblem = NonlinearVariationalProblem(F, q, bcs=BCs)
forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

# Initialise counters and solve numerically
t = 0.
cnt = 0
forwardFile.write(u_, eta_, time=t)
print('Generating numerical solution')
while t < op.Tend:
    forwardSolver.solve()
    q_.assign(q)
    if cnt % op.ndump == 0:
        forwardFile.write(u_, eta_, time=t)
        print('t = %.1fs' % t)
    t += op.dt
    cnt += 1

# # Plot analytic solution
# t = 0.
# print('Generating analytical solution')
# while t < op.Tend:
#     q = form.solutionHuang(V, t=t)
#     u, eta = q.split()
#     u.rename('Analytic fluid velocity')
#     eta.rename('Analytic free surface')
#     solFile.write(u, eta, time=t)
#     print('t = %.1fs' % t)
#     t += op.ndump * op.dt
