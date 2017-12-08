from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt

print('\n***************** EQUATORIAL ROSSBY WAVE TEST PROBLEM *********************\n')
print('Mesh adaptive solver defined on a rectangular mesh')
approach = input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': ") or 'goalBased'

# Cheat code to resume from saved data in goalBased case
if approach == 'saved':
    approach = 'goalBased'
    getData = False
else:
    getData = True
useAdjoint = approach == 'goalBased'

# Establish filenames
dirName = "plots/testSuite/"
forwardFile = File(dirName + "forwardRW.pvd")
residualFile = File(dirName + "residualRW.pvd")
adjointFile = File(dirName + "adjointRW.pvd")
errorFile = File(dirName + "errorIndicatorRW.pvd")
adaptiveFile = File(dirName + "goalBasedRW.pvd") if useAdjoint else File(dirName + "simpleAdaptRW.pvd")

# Specify physical and solver parameters
op = opt.Options(dt=0.05, Tstart=0.5, Tend=2.5, family='dg-cg',
                 hmin=5e-2, hmax=1., rm=5, gradate=False, advect=False,
                 vscale=0.4 if useAdjoint else 0.85)
dt = op.dt
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
b = Constant(0.1)
op.checkCFL(b)

# Define inital mesh and FunctionSpace
n = 32
N = 2 * n
lx = 2 * np.pi
mesh = SquareMesh(n, n, lx, lx)   # Computational mesh
mesh_N = SquareMesh(N, N, lx, lx)   # Finer mesh (N > n) upon which to approximate error
x, y = SpatialCoordinate(mesh)
V_n = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
V_N = VectorFunctionSpace(mesh_N, op.space1, op.degree1) * FunctionSpace(mesh_N, op.space2, op.degree2)

if useAdjoint:
    rho = Function(V_N)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    dual = Function(V_n)
    dual_u, dual_e = dual.split()
    dual_u.rename("Adjoint velocity")
    dual_e.rename("Adjoint elevation")
    P0_N = FunctionSpace(mesh_N, "DG", 0)
    v = TestFunction(P0_N)

# Apply initial condition and define Functions
ic = project(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V_n.sub(1))  # TODO: alter IC
q_ = Function(V_n)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.assign(ic)
q = Function(V_n)
u, eta = q.split()
u.rename("Velocity")
eta.rename("Elevation")

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if getData:
    if approach in ('fixedMesh', 'goalBased'):
        # Define variational problem
        qt = TestFunction(V_n)
        forwardProblem = NonlinearVariationalProblem(form.weakResidualMSW(q, q_, qt, Constant(1), Dt, y), q)
        forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

        print('\nStarting fixed mesh primal run (forwards in time)')
        finished = False
        primalTimer = clock()
        forwardFile.write(u, eta, time=t)
        while t < T:
            # Solve problem at current timestep
            forwardSolver.solve()

            # Approximate residual of forward equation and save to HDF5
            if useAdjoint:
                if not cnt % rm:
                    qN, q_N = inte.mixedPairInterp(mesh_N, V_N, q, q_)
                    Au, Ae = form.strongResidualSW(qN, q_N, b, Dt)
                    rho_u.interpolate(Au)
                    rho_e.interpolate(Ae)
                    with DumbCheckpoint(dirName + 'hdf5/residual_RW' + op.indexString(cnt), mode=FILE_CREATE) as chk:
                        chk.store(rho_u)
                        chk.store(rho_e)
                        chk.close()
                    residualFile.write(rho_u, rho_e, time=t)

            # Update solution at previous timestep
            q_.assign(q)

            # Mark timesteps to be used in adjoint simulation
            if useAdjoint:
                if t >= T - dt:
                    finished = True
                if t == 0.:
                    adj_start_timestep()
                else:
                    adj_inc_timestep(time=t, finished=finished)

            forwardFile.write(u, eta, time=t)
            print('t = %.3fs' % t)
            t += dt
            cnt += 1
        cnt -= 1
        primalTimer = clock() - primalTimer
        print('Primal run complete. Run time: %.3fs' % primalTimer)
