from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor

print('\n*********************** SHALLOW WATER TEST PROBLEM ************************\n')
print('Mesh adaptive solver initially defined on a square mesh')
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
forwardFile = File(dirName + "forwardSW.pvd")
residualFile = File(dirName + "residualSW.pvd")
adjointFile = File(dirName + "adjointSW.pvd")
errorFile = File(dirName + "errorIndicatorSW.pvd")
adaptiveFile = File(dirName + "goalBasedSW.pvd") if useAdjoint else File(dirName + "simpleAdaptSW.pvd")

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
ic = project(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))), V_n.sub(1))
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
        forwardProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
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
                    with DumbCheckpoint(dirName + 'hdf5/residual_SW' + stor.indexString(cnt), mode=FILE_CREATE) as chk:
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

    if useAdjoint:
        # Set up adjoint problem
        J = form.objectiveFunctionalSW(q, Tstart=Ts, x1=0., x2=np.pi/2, y1=0.5*np.pi, y2=1.5*np.pi, smooth=False)
        parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
        t = T
        save = True

        # Time integrate (backwards)
        print('\nStarting fixed mesh dual run (backwards in time)')
        dualTimer = clock()
        for (variable, solution) in compute_adjoint(J):
            if save:
                # Load adjoint data. NOTE the interpolation operator is overloaded
                dual_u.dat.data[:] = variable.dat.data[0]
                dual_e.dat.data[:] = variable.dat.data[1]
                dual_N = inte.mixedPairInterp(mesh_N, V_N, dual)[0]

                if not cnt % rm:
                    indexStr = stor.indexString(cnt)

                    # Load residual data from HDF5
                    with DumbCheckpoint(dirName + 'hdf5/residual_SW' + indexStr, mode=FILE_READ) as loadResidual:
                        loadResidual.load(rho_u)
                        loadResidual.load(rho_e)
                        loadResidual.close()

                    # Estimate error using forward residual (DWR)
                    epsilon = assemble(v * inner(rho, dual_N) * dx)
                    epsNorm = np.abs(assemble(inner(rho, dual_N) * dx))   # Normalise
                    if epsNorm == 0.:
                        epsNorm = 1.
                    epsilon.dat.data[:] = np.abs(epsilon.dat.data) / epsNorm
                    epsilon.rename("Error indicator")

                    # Save error indicator data to HDF5
                    with DumbCheckpoint(dirName + 'hdf5/error_SW' + indexStr, mode=FILE_CREATE) as saveError:
                        saveError.store(epsilon)
                        saveError.close()

                    # Print to screen, save data and increment counters
                    errorFile.write(epsilon, time=t)
                adjointFile.write(dual_u, dual_e, time=t)
                print('t = %.3fs' % t)
                t -= dt
                cnt -= 1
                save = False
            else:
                save = True
            if (cnt == -1) | (t < -dt):
                break
        dualTimer = clock() - dualTimer
        print('Adjoint run complete. Run time: %.3fs' % dualTimer)
        t += dt
        cnt += 1

        # Reset initial conditions for primal problem
        u_.interpolate(Expression([0, 0]))
        eta_.assign(ic)
        epsilon = Function(P0_N, name="Error indicator")

if approach in ('simpleAdapt', 'goalBased'):
    print('\nStarting adaptive mesh primal run (forwards in time)')
    adaptTimer = clock()
    while t <= T:
        if not cnt % rm:
            stepTimer = clock()

            # Reconstruct Hessian
            W = TensorFunctionSpace(mesh, "CG", 1)
            H = adap.constructHessian(mesh, W, eta, op=op)

            # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
            if useAdjoint:
                with DumbCheckpoint(dirName + 'hdf5/error_SW' + stor.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon)                 # P0 field on the initial mesh
                    loadError.close()
                errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
                for k in range(mesh.topology.num_vertices()):
                    H.dat.data[k] *= errEst.dat.data[k]     # Scale by error estimate

            # Adapt mesh and interpolate variables
            M = adap.computeSteadyMetric(mesh, W, H, eta, nVerT=nVerT, op=op)
            if op.gradate:
                adap.metricGradation(mesh, M)
            if op.advect:
                M = adap.advectMetric(M, u, Dt, n=rm)
            mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
            V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
            q_ = inte.mixedPairInterp(mesh, V, q_)[0]
            q = Function(V)
            u, eta = q.split()
            u.rename("Velocity")
            eta.rename("Elevation")

            # Re-establish variational form
            qt = TestFunction(V)
            adaptProblem = NonlinearVariationalProblem(form.weakResidualSW(q, q_, qt, b, Dt), q)
            adaptSolver = NonlinearVariationalSolver(adaptProblem, solver_parameters=op.params)

            # Get mesh stats
            nEle = msh.meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt / rm + 1, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, mM)

        # Solve problem at current timestep
        adaptSolver.solve()
        q_.assign(q)

        # Print to screen, save data and increment counters
        print('t = %.3fs' % t)
        adaptiveFile.write(u, eta, time=t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)

# Print to screen timing analyses (and error in pure advection case)
if getData and useAdjoint:
    print("TIMINGS:         Forward run   %5.3fs, Adjoint run   %5.3fs, Adaptive run   %5.3fs" %
          (primalTimer, dualTimer, adaptTimer))
