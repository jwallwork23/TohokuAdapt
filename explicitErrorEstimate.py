from firedrake import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.domain as dom
import utils.interpolation as inte
import utils.options as opt
import utils.storage as stor


# Define initial mesh and mesh statistics placeholders
# print('************** ADJOINT-BASED ADAPTIVE TSUNAMI SIMULATION **************\n')
# print('ADJOINT-GUIDED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 4): ') or 4))
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
mesh0 = mesh
W0 = VectorFunctionSpace(mesh0, 'CG', 2) * FunctionSpace(mesh0, 'CG', 1)
print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d.' % (nEle, nVer))

# Get default parameter values and check CFL criterion
op = opt.Options(vscale=0.2, rm=60)
nVerT = op.vscale * nVer    # Target #Vertices
iso = op.iso
dirName = 'plots/adjointBased/explicit/'
if iso:
    dirName += 'isotropic/'
g = op.g
T = op.T
dt = op.dt
Dt = Constant(dt)
op.checkCFL(b)
ndump = op.ndump
rm = op.rm
stored = bool(input('Hit anything but enter if adjoint data is already stored: '))
#
# if not stored:
#     # Initalise counters and forcing switch
#     t = T
#     mn = int(T / (rm * dt))
#     dumpn = ndump
#     meshn = rm
#     tic1 = clock()
#     coeff = Constant(1.)
#     switch = True
#
#     # Establish adjoint variables and apply initial conditions
#     lam_ = Function(W0)
#     lu_, le_ = lam_.split()
#     lu_.interpolate(Expression([0, 0]))
#     le_.interpolate(Expression(0))
#     lam = Function(W0).assign(lam_)
#     lu, le = lam.split()
#     lu.rename('Adjoint velocity')
#     le.rename('Adjoint free surface')
#
#     # Store final time data to HDF5 and PVD
#     with DumbCheckpoint(dirName + 'hdf5/adjoint_' + stor.indexString(mn), mode=FILE_CREATE) as chk:
#         chk.store(lu)
#         chk.store(le)
#         chk.close()
#     adjointFile = File(dirName + 'adjoint.pvd')
#     adjointFile.write(lu, le, time=T)
#
#     # Establish (smoothened) indicator function for adjoint equations
#     fexpr = '(x[0] > 490e3) & (x[0] < 640e3) & (x[1] > 4160e3) & (x[1] < 4360e3) ? ' \
#             'exp(1. / (pow(x[0] - 565e3, 2) - pow(75e3, 2))) * exp(1. / (pow(x[1] - 4260e3, 2) - pow(100e3, 2))) : 0.'
#     f = Function(W0.sub(1), name='Forcing term').interpolate(Expression(fexpr))
#
#     # Set up the variational problem, using Crank Nicolson timestepping
#     w, xi = TestFunctions(W0)
#     lu, le = split(lam)
#     lu_, le_ = split(lam_)
#     L = ((le - le_) * xi + inner(lu - lu_, w)
#          - Dt * op.g * inner(0.5 * (lu + lu_), grad(xi)) - coeff * f * xi
#          + Dt * (b * inner(grad(0.5 * (le + le_)), w) + 0.5 * (le + le_) * inner(grad(b), w))) * dx
#     adjointProblem = NonlinearVariationalProblem(L, lam)
#     adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=op.params)
#     lu, le = lam.split()
#     lu_, le_ = lam_.split()
#
#     print('\nStarting fixed resolution adjoint run...')
#     while t > 0.5 * dt:
#         t -= dt
#         dumpn -= 1
#         meshn -= 1
#
#         # Modify forcing term
#         if (t < op.Ts + 1.5 * dt) & switch:
#             coeff.assign(0.5)
#         elif (t < op.Ts + 0.5 * dt) & switch:
#             switch = False
#             coeff.assign(0.)
#
#         # Solve the problem, update variables and dump to vtu and HDF5
#         adjointSolver.solve()
#         lam_.assign(lam)
#         if dumpn == 0:
#             dumpn += ndump
#             adjointFile.write(lu, le, time=t)
#         if meshn == 0:
#             meshn += rm
#             mn -= 1
#             if not stored:
#                 print('t = %1.1fs' % t)
#                 with DumbCheckpoint(dirName + 'hdf5/adjoint_' + stor.indexString(mn), mode=FILE_CREATE) as chk:
#                     chk.store(lu)
#                     chk.store(le)
#                     chk.close()
#     toc1 = clock()
#     print('... done! Elapsed time for adjoint solver: %1.2fs' % (toc1 - tic1))


t = 0.


# Initialise variables and specify bathymetry
q_ = Function(W0)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(eta0)
q = Function(W0).assign(q_)
u, eta = q.split()
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Establish variational problem
v, ze = TestFunctions(W0)
u, eta = split(q)
u_, eta_ = split(q_)
uh = 0.5 * (u + u_)
etah = 0.5 * (eta + eta_)
L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze))
     + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
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
rk_01 = Function(W0.sub(0), name="Element residual xy")
rk_2 = Function(W0.sub(1), name="Element residual z")
rb = Function(W0.sub(1), name="Boundary residual")
hk = Function(W0.sub(1), name="Element size").interpolate(CellSize(mesh))
qfile = File(dirName + 'shallowWater.pvd')
qfile.write(u, eta, time=t)
rfile = File(dirName + 'residualSW.pvd')

# DG test functions to get cell-wise norms
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)
n = FacetNormal(mesh)

while t < T - 0.5 * dt:
    t += dt
    print('t = %5.2fs' % t)
    pde_solve.solve()

    # TODO this in Thetis, we will need to load TWO timesteps worth of field data

    # Get element residual
    rk_01.interpolate(u_ - u - Dt * g * grad(etah))
    rk_2.interpolate(eta_ - eta - Dt * div(b * uh))
    rho = assemble(v * sqrt(dot(rk_01, rk_01) + rk_2 * rk_2) / CellVolume(mesh) * dx)

    # Get boundary residual     TODO: this only currently integrates over domain the boundary, NOT cell boundaries
    rho += assemble(v * Dt * b * dot(uh, n) * ds)
    rho.rename('Local error indicators')

    # Update variables and output (locally constant) error indicators
    q_.assign(q)
    qfile.write(u, eta, time=t)
    rfile.write(rho, time=t)
