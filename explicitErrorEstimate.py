from firedrake import *
import numpy as np

import utils.error as err


# Set parameter values
depth = 0.1
T = 2.5
g = 9.81
dt = 0.05
Dt = Constant(dt)

# Define mesh and function space
n = 32
lx = 4
mesh = SquareMesh(lx * n, lx * n, lx, lx)
x, y = SpatialCoordinate(mesh)
W = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "CG", 2)
b = Function(W.sub(1), name="Bathymetry").assign(depth)

# Initalise counters
t = T
mn = int(T / dt)

# Solver parameters
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Establish adjoint variables and apply initial conditions
lam_ = Function(W)
lu_, le_ = lam_.split()
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression(0))
lam = Function(W).assign(lam_)
lu, le = lam.split()
lu.rename("Adjoint velocity")
le.rename("Adjoint free surface")

# Store final time data to HDF5 and PVD
with DumbCheckpoint("plots/adjointBased/explicit/hdf5/adjoint_" + str(mn), mode=FILE_CREATE) as chk:
    chk.store(lu)
    chk.store(le)
    chk.close()
adjointFile = File("plots/adjointBased/explicit/adjoint.pvd")
adjointFile.write(lu, le, time=T)

# Establish (smoothened) indicator function for adjoint equations
fexpr = "(x[0] >= 0.) & (x[0] < 0.25) & (x[1] > 1.8) & (x[1] < 2.2) ? 1e-3 : 0."
f = Function(W.sub(1), name="Forcing term").interpolate(Expression(fexpr))

# Set up the variational problem, using Crank Nicolson timestepping
w, xi = TestFunctions(W)
lu, le = split(lam)
lu_, le_ = split(lam_)
L = ((le - le_) * xi + inner(lu - lu_, w)
     - Dt * g * inner(0.5 * (lu + lu_), grad(xi)) - f * xi
     + Dt * (b * inner(grad(0.5 * (le + le_)), w) + 0.5 * (le + le_) * inner(grad(b), w))) * dx
adjointProblem = NonlinearVariationalProblem(L, lam)
adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=params)
lu, le = lam.split()
lu_, le_ = lam_.split()

while mn > 0:
    t -= dt
    mn -= 1
    print("mn = %3d, t = %5.2fs" % (mn, t))

    # Solve the problem, update variables and dump to vtu and HDF5
    adjointSolver.solve()
    lam_.assign(lam)
    adjointFile.write(lu, le, time=t)
    with DumbCheckpoint("plots/adjointBased/explicit/hdf5/adjoint_" + str(mn), mode=FILE_CREATE) as chk:
        chk.store(lu)
        chk.store(le)
        chk.close()

assert(mn == 0)

# Initialise variables and specify bathymetry
q_ = Function(W)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
q = Function(W).assign(q_)
u, eta = q.split()
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Establish variational problem
v, ze = TestFunctions(W)
u, eta = split(q)
u_, eta_ = split(q_)
uh = 0.5 * (u + u_)
etah = 0.5 * (eta + eta_)
L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
pde = NonlinearVariationalProblem(L, q)
pde_solve = NonlinearVariationalSolver(pde, solver_parameters=params)
u_, eta_ = q_.split()
u, eta = q.split()

# Set up output files
qfile = File("plots/adjointBased/explicit/forward.pvd")
qfile.write(u, eta, time=t)
rfile = File("plots/adjointBased/explicit/residualEstimate.pvd")

while mn < int(T / dt):
    t += dt
    mn += 1
    print("mn = %3d, t = %5.2fs" % (mn, t))
    pde_solve.solve()

    # Load adjoint data and compute local error indicators
    with DumbCheckpoint("plots/adjointBased/explicit/hdf5/adjoint_" + str(mn), mode=FILE_READ) as chk:
        chk.load(lu, name="Adjoint velocity")
        chk.load(le, name="Adjoint free surface")
        chk.close()
    rho = err.explicitErrorEstimator(W, u_, u, eta_, eta, lu, le, b, dt)
    q_.assign(q)
    qfile.write(u, eta, time=t)
    rfile.write(rho, time=t)
