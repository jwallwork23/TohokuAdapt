from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.bootstrapping as boot
import utils.error as err
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt

print('\n***************** EQUATORIAL ROSSBY WAVE TEST PROBLEM *********************\n')
print('Mesh adaptive solver defined on a rectangular mesh')
approach, getData, getError = msc.cheatCodes(input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': "))
useAdjoint = approach == 'goalBased'
periodic = True

# Establish filenames
dirName = "plots/testSuite/"
forwardFile = File(dirName + "forwardRW.pvd")
residualFile = File(dirName + "residualRW.pvd")
adjointFile = File(dirName + "adjointRW.pvd")
errorFile = File(dirName + "errorIndicatorRW.pvd")
adaptiveFile = File(dirName + "goalBasedRW.pvd") if useAdjoint else File(dirName + "simpleAdaptRW.pvd")

# TODO: bootstrapping
n = 4

# Define initial mesh and FunctionSpace
N = 2 * n
lx = 48
ly = 24
if periodic:
    mesh = PeriodicRectangleMesh(lx * n, ly * n, lx, ly, direction="x")     # Computational mesh
    mesh_N = PeriodicRectangleMesh(lx * N, ly * N, lx, ly, direction="x")
else:
    mesh = RectangleMesh(lx * n, ly * n, lx, ly)    # Computational mesh
    mesh_N = RectangleMesh(lx * N, ly * N, lx, ly)
for m in (mesh, mesh_N):
    xy = Function(m.coordinates)
    xy.dat.data[:, 0] -= 24.
    xy.dat.data[:, 1] -= 12.
    m.coordinates.assign(xy)
x, y = SpatialCoordinate(mesh)

# Define FunctionSpaces and specify physical and solver parameters
op = opt.Options(Tstart=0,
                 Tend=120,
                 family='dg-cg',
                 hmin=0.2,
                 hmax=4.,
                 rm=5,
                 gradate=False,
                 advect=False,
                 window=True if useAdjoint else False,
                 vscale=0.4 if useAdjoint else 0.85,
                 plotpvd=True if getData == False else False)
V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
V_N = VectorFunctionSpace(mesh_N, op.space1, op.degree1) * FunctionSpace(mesh_N, op.space2, op.degree2)
b = Constant(1.)
h = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
dt = adap.adaptTimestepSW(mesh, b)
print('     #### Using initial timestep = %4.3fs\n' % dt)
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
ndump = op.ndump

# Define Functions relating to goalBased approach
if useAdjoint:
    rho = Function(V_N)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_N = Function(V_N)
    dual_N_u, dual_N_e = dual_N.split()
    dual_N_u.rename("Adjoint velocity")
    dual_N_e.rename("Adjoint elevation")
    P0_N = FunctionSpace(mesh_N, "DG", 0)
    v = TestFunction(P0_N)
    epsilon = Function(P0_N, name="Error indicator")

# Apply initial and boundary conditions
q_ = form.analyticHuang(V)
u_, eta_ = q_.split()
q = Function(V)
q.assign(q_)
u, eta = q.split()
u.rename("Velocity")
eta.rename("Elevation")
if periodic:
    bc = DirichletBC(V.sub(0), [0, 0], [1, 2])              # No-slip on top and bottom of domain
else:
    bc = DirichletBC(V_n.sub(0), [0, 0], 'on_boundary')     # No-slip on entire boundary

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iStart = int(op.Tstart / dt)
iEnd = np.ceil(T / dt)
nEle, nVer = msh.meshStats(mesh)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if getData:
    # Define variational problem
    qt = TestFunction(V)
    F = form.weakResidualSW(q, q_, qt, b, Dt, g=1., f0=0., beta=1.,
                            rotational=True, nonlinear=False, allowNormalFlow=True)
    forwardProblem = NonlinearVariationalProblem(F, q, bcs=bc)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    print('\nStarting fixed mesh primal run (forwards in time)')
    finished = False
    primalTimer = clock()
    forwardFile.write(u, eta, time=t)
    while t < T:
        # Solve problem at current timestep
        forwardSolver.solve()

        # # Approximate residual of forward equation and save to HDF5
        # if useAdjoint:
        #     if not cnt % rm:
        #         qN, q_N = inte.mixedPairInterp(mesh_N, V_N, q, q_)
        #         Au, Ae = form.strongResidualSW(qN, q_N, b, Dt)
        #         rho_u.interpolate(Au)
        #         rho_e.interpolate(Ae)
        #         with DumbCheckpoint(dirName + 'hdf5/residual_RW' + op.indexString(cnt), mode=FILE_CREATE) as chk:
        #             chk.store(rho_u)
        #             chk.store(rho_e)
        #             chk.close()
        #         residualFile.write(rho_u, rho_e, time=t)

        # Update solution at previous timestep
        q_.assign(q)

        # # Mark timesteps to be used in adjoint simulation
        # if useAdjoint:
        #     if t >= T - dt:
        #         finished = True
        #     if t == 0.:
        #         adj_start_timestep()
        #     else:
        #         adj_inc_timestep(time=t, finished=finished)

        if cnt % ndump == 0:
            forwardFile.write(u, eta, time=t)
            print('t = %.2fs' % t)
        t += dt
        cnt += 1
    cnt -= 1
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)
