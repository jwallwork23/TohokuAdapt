from firedrake import *
import numpy as np

# Set parameter values
depth = 0.1
T = 2.5
g = 9.81
dt = 0.05
Dt = Constant(dt)
t = 0.

# Define mesh and function space
n = 16
lx = 4
mesh = SquareMesh(lx * n, lx * n, lx, lx)
x, y = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, "CG", 2) * FunctionSpace(mesh, "CG", 1)   # Taylor-Hood mixed space

# Initialise variables and specify bathymetry
q_ = Function(V)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
q = Function(V).assign(q_)
u, eta = q.split()
u.rename("Fluid velocity")
eta.rename("Free surface displacement")
b = Function(V.sub(1), name="Bathymetry")
b.assign(depth)

# Establish variational problem
v, ze = TestFunctions(V)
u, eta = split(q)
u_, eta_ = split(q_)
uh = 0.5 * (u + u_)
etah = 0.5 * (eta + eta_)
L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
pde = NonlinearVariationalProblem(L, q)
pde_solve = NonlinearVariationalSolver(pde, solver_parameters={'mat_type': 'matfree',
                                                               'snes_type': 'ksponly',
                                                               'pc_type': 'python',
                                                               'pc_python_type': 'firedrake.AssembledPC',
                                                               'assembled_pc_type': 'lu',
                                                               'snes_lag_preconditioner': -1,
                                                               'snes_lag_preconditioner_persists': True})
u_, eta_ = q_.split()
u, eta = q.split()

# Set up auxiliary functions and output files
rk_01 = Function(V.sub(0), name="Element residual xy")
rk_2 = Function(V.sub(1), name="Element residual z")
rb = Function(V.sub(1), name="Boundary residual")
hk = Function(V.sub(1), name="Element size").interpolate(CellSize(mesh))
qfile = File('plots/shallowWater.pvd')
qfile.write(u, eta, time=t)
rfile = File('plots/residualSW.pvd')

# DG test functions to get cell-wise norms
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)
n = FacetNormal(mesh)

while t < T - 0.5 * dt:
    t += dt
    print('t = %5.2fs' % t)
    pde_solve.solve()

    # Get element residual
    rk_01.interpolate(u_ - u - Dt * g * grad(etah))
    rk_2.interpolate(eta_ - eta - Dt * div(b * uh))
    rho = assemble(v * sqrt(dot(rk_01, rk_01) + rk_2 * rk_2) / CellVolume(mesh) * dx)

    # Get boundary residual     TODO: this only currently integrates over domain the boundary, NOT cell boundaries
    rho += assemble(Dt * b * dot(uh, n) * ds)
    rho.rename('Local error indicators')

    # Update variables and output (locally constant) error indicators
    q_.assign(q)
    qfile.write(u, eta, time=t)
    rfile.write(rho, time=t)
