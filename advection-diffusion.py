from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.misc as msc
import utils.options as opt

print('\n******************************** ADVECTION-DIFFUSION TEST PROBLEM ********************************\n')
print('Mesh adaptive solver initially defined on a rectangular mesh')
approach = input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': ") or 'goalBased'
diffusion = True

# Cheat code to resume from saved data in goalBased case
if approach == 'saved':
    approach = 'goalBased'
    getData = False
else:
    getData = True
useAdjoint = approach == 'goalBased'

# Establish filenames
dirName = "plots/testSuite/"
forwardFile = File(dirName + "forwardAD.pvd")
residualFile = File(dirName + "residualAD.pvd")
adjointFile = File(dirName + "adjointAD.pvd")
errorFile = File(dirName + "errorIndicatorAD.pvd")
adaptiveFile = File(dirName + "goalBasedAD.pvd") if useAdjoint else File(dirName + "simpleAdaptAD.pvd")
metricFile = File(dirName + "metricAD.pvd")

# Define Mesh and FunctionSpace
n = 16
N = 2 * n
mesh_n = RectangleMesh(4 * n, n, 4, 1)  # Computational mesh
mesh_N = RectangleMesh(4 * N, N, 4, 1)  # Finer mesh (N > n) upon which to approximate error
x, y = SpatialCoordinate(mesh_n)
V_n = FunctionSpace(mesh_n, "CG", 2)
V_N = FunctionSpace(mesh_N, "CG", 2)

if useAdjoint:
    dual_n = Function(V_n, name='Adjoint')
    rho_N = Function(V_N, name='Residual')
    P0_N = FunctionSpace(mesh_N, "DG", 0)
    v = TestFunction(P0_N)

# Specify physical and solver parameters
op = opt.Options(dt=0.04,
                 Tend=2.4,
                 hmin=5e-2,
                 hmax=0.8,
                 rm=5,
                 gradate=False,
                 advect=False,
                 vscale=0.4 if useAdjoint else 0.85)
dt = op.dt
Dt = Constant(dt)
T = op.Tend
w = Function(VectorFunctionSpace(mesh_n, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
nu = 1e-3 if diffusion else 0.

# Apply initial condition and define Functions
ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V_n)
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
phi_next = Function(V_n, name='Concentration next')

# Analytic solution in pure advection case
if not diffusion:
    fc = project(exp(- (pow(x - 0.5 - T, 2) + pow(y - 0.5, 2)) / 0.04), V_n)  # Final value in pure advection case
    File(dirName + "finalValueAD.pvd").write(fc)

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh_n)
mM = [nEle, nEle]            # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
t = 0.
cnt = 0

if getData:
    if approach in ('fixedMesh', 'goalBased'):
        # Define variational problem
        psi = TestFunction(V_n)
        F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
        bc = DirichletBC(V_n, 0., "on_boundary")

        print('\nStarting fixed mesh primal run (forwards in time)')
        finished = False
        primalTimer = clock()
        while t < T:
            # Solve problem at current timestep
            solve(F == 0, phi_next, bc)

            # Approximate residual of forward equation and save to HDF5
            if useAdjoint:
                if not cnt % rm:
                    phi_next_N, phi_N, w_N = inte.interp(mesh_N, phi_next, phi, w)
                    rho_N.interpolate(form.strongResidualAD(phi_next_N, phi_N, w_N, Dt, nu=nu))
                    with DumbCheckpoint(dirName + 'hdf5/residual_AD' + msc.indexString(cnt), mode=FILE_CREATE) as saveRes:
                        saveRes.store(rho_N)
                        saveRes.close()
                    # residualFile.write(rho_N, time=t)

            # Update solution at previous timestep
            phi.assign(phi_next)

            # Mark timesteps to be used in adjoint simulation
            if useAdjoint:
                if t > T:
                    finished = True
                if t == 0.:
                    adj_start_timestep()
                else:
                    adj_inc_timestep(time=t, finished=finished)

            # forwardFile.write(phi, time=t)
            print('t = %.3fs' % t)
            t += dt
            cnt += 1
        cnt -= 1
        primalTimer = clock() - primalTimer
        print('Primal run complete. Run time: %.3fs' % primalTimer)
        finalFixed = Function(V_n)
        finalFixed.dat.data[:] = phi.dat.data   # NOTE assign, copy and interpolate functions are overloaded

    if useAdjoint:
        # Set up adjoint problem
        J = form.objectiveFunctionalAD(phi)
        parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
        t = T
        save = True

        # Time integrate (backwards)
        print('\nStarting fixed mesh dual run (backwards in time)')
        dualTimer = clock()
        for (variable, solution) in compute_adjoint(J):
            if save:
                # Load adjoint data. NOTE the interpolation operator is overloaded
                dual_n.dat.data[:] = variable.dat.data
                dual_N = inte.interp(mesh_N, dual_n)[0]

                if not cnt % rm:
                    indexStr = msc.indexString(cnt)

                    # Load residual data from HDF5
                    with DumbCheckpoint(dirName + 'hdf5/residual_AD' + indexStr, mode=FILE_READ) as loadRes:
                        loadRes.load(rho_N)
                        loadRes.close()

                    # Estimate error using forward residual
                    epsilon_N = assemble(v * rho_N * dual_N * dx)   # Currently a P0 field
                    epsNorm = np.abs(assemble(rho_N * dual_N * dx))
                    if epsNorm == 0.:
                        epsNorm = 1.
                    epsilon_N.dat.data[:] = np.abs(epsilon_N.dat.data) / epsNorm
                    epsilon_N.rename("Error indicator")

                    # Save error indicator data to HDF5
                    with DumbCheckpoint(dirName + 'hdf5/error_AD' + indexStr, mode=FILE_CREATE) as saveError:
                        saveError.store(epsilon_N)
                        saveError.close()

                    # Print to screen, save data and increment counters
                    errorFile.write(epsilon_N, time=t)
                # adjointFile.write(dual_n, time=t)
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

        # Reset initial conditions for primal problem and recreate error indicator placeholder
        phi = ic.copy(deepcopy=True)
        phi.rename('Concentration')

if approach in ('simpleAdapt', 'goalBased'):
    print('\nStarting adaptive mesh primal run (forwards in time)')

    adaptTimer = clock()
    if useAdjoint:
        epsilon_N = Function(P0_N, name="Error indicator")

    while t <= T:
        if not cnt % rm:
            stepTimer = clock()

            # Construct metric
            W = TensorFunctionSpace(mesh_n, "CG", 1)

            if useAdjoint & (cnt != 0):

                # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
                with DumbCheckpoint(dirName + 'hdf5/error_AD' + msc.indexString(cnt), mode=FILE_READ) as loadError:
                    loadError.load(epsilon_N)
                    loadError.close()
                errEst = Function(FunctionSpace(mesh_n, "CG", 1)).interpolate(inte.interp(mesh_n, epsilon_N)[0])
                # errEst.dat.data[:] *= 2000
                M = adap.isotropicMetric(W, errEst, op=op, invert=False)

                # TODO: what is the best way to do this?

            else:
                H = adap.constructHessian(mesh_n, W, phi, op=op)
                M = adap.computeSteadyMetric(mesh_n, W, H, phi, nVerT=nVerT, op=op)
            M.rename("Metric")
            metricFile.write(M)

            # Adapt mesh and interpolate variables
            if op.gradate:
                adap.metricGradation(mesh_n, M)
            if op.advect:
                M = adap.advectMetric(M, w, Dt, n=rm, fieldToAdvect='M')
            mesh_n = AnisotropicAdaptation(mesh_n, M).adapted_mesh
            phi = inte.interp(mesh_n, phi)[0]
            phi.rename("Concentration")
            V_n = FunctionSpace(mesh_n, "CG", 2)
            phi_next = Function(V_n, name="Concentration next")

            # Re-establish bilinear form and set boundary conditions
            psi = TestFunction(V_n)
            w = Function(VectorFunctionSpace(mesh_n, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
            F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
            bc = DirichletBC(V_n, 0., "on_boundary")

            # Get mesh stats
            nEle = msh.meshStats(mesh_n)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            op.printToScreen(cnt / rm + 1, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, mM)

        # Solve problem at current timestep
        solve(F == 0, phi_next, bc)
        phi.assign(phi_next)

        # Print to screen, save data and increment counters
        print('t = %.3fs' % t)
        adaptiveFile.write(phi, time=t)
        t += dt
        cnt += 1
    adaptTimer = clock() - adaptTimer
    print('Adaptive primal run complete. Run time: %.3fs \n' % adaptTimer)
    finalAdapt = phi.copy(deepcopy=True)

# Print to screen timing analyses (and error in pure advection case)
if getData and useAdjoint:
    print("TIMINGS:         Forward run   %5.3fs, Adjoint run   %5.3fs, Adaptive run   %5.3fs" %
          (primalTimer, dualTimer, adaptTimer))
    if not diffusion:
        print("RELATIVE ERRORS: Fixed mesh run   %5.3f, Adaptive run   %5.3f"
              % (errornorm(finalFixed, fc) / norm(fc), errornorm(finalAdapt, inte.interp(mesh_n, fc)[0]) / norm(fc)))
