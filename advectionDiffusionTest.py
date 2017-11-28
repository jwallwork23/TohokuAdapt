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

print('\n******************************** ADVECTION-DIFFUSION TEST PROBLEM ********************************\n')
print('Mesh adaptive solver initially defined on a rectangular mesh')
useAdjoint = bool(input("Hit anything except enter to use adjoint equations to guide adaptive process. "))

# Establish filenames
dirName = "plots/advectionDiffusion/"
forwardFile = File(dirName + "forwardAD.pvd")
residualFile = File(dirName + "residualAD.pvd")
adjointFile = File(dirName + "adjointAD.pvd")
errorFile = File(dirName + "errorIndicatorAD.pvd")
adaptiveFile = File(dirName + "goalBasedAD.pvd") if useAdjoint else File(dirName + "simpleAdaptAD.pvd")

# Define Mesh and FunctionSpace
n = 16
mesh = RectangleMesh(4 * n, n, 4, 1)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 2)
P0 = FunctionSpace(mesh, "DG", 0)
v = TestFunction(P0)

# Specify and apply initial condition
ic = project(exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04), V)
phi = ic.copy(deepcopy=True)
phi.rename('Concentration')
phi_next = Function(V, name='Concentration next')
psi = TestFunction(V)
dual = Function(V, name='Adjoint')
rho = Function(V, name='Residual')

# Specify physical and solver parameters
op = opt.Options(dt=0.04, T=2.4, hmin=5e-2, hmax=1., rm=5, vscale=0.4 if useAdjoint else 0.85, gradate=False,
                 advect=True)
dt = op.dt
Dt = Constant(dt)
w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
nu = 1e-3   # Diffusivity

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]            # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale

# Establish bilinear form and set boundary conditions
F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
bc = DirichletBC(V, 0., "on_boundary")

# Initialise counters
t = 0.
T = op.T
cnt = 0

if useAdjoint:
    print('Starting fixed mesh primal run (forwards in time)')
    finished = False
    primalTimer = clock()
    while t <= T:
        # Solve problem at current timestep
        solve(F == 0, phi_next, bc)
        phi.assign(phi_next)

        # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
        if t >= T - dt:
            finished = True
        if t == 0.:
            adj_start_timestep()
        else:
            adj_inc_timestep(time=t, finished=finished)

        # Approximate residual of forward equation and save to HDF5
        rho.interpolate(form.strongResidualAD(phi_next, phi, w, Dt, nu=nu))
        with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_CREATE) as saveResidual:
            saveResidual.store(rho)
            saveResidual.close()

        # Print to screen, save data and increment counters
        print('t = %.3fs' % t)
        forwardFile.write(phi, time=t)
        residualFile.write(rho, time=t)
        t += dt
        cnt += 1
    cnt -= 1
    primalTimer = clock() - primalTimer
    print('Primal run complete. Run time: %.3fs' % primalTimer)

    # Set up adjoint problem
    J = form.objectiveFunctionalAD(phi)
    parameters["adjoint"]["stop_annotating"] = True     # Stop registering equations
    t = T
    save = True

    # Time integrate (backwards)
    print('Starting fixed mesh dual run (backwards in time)')
    dualTimer = clock()
    for (variable, solution) in compute_adjoint(J):
        try:
            dual.dat.data[:] = variable.dat.data
            if save:
                # Load residual data from HDF5
                with DumbCheckpoint(dirName + 'hdf5/residual_' + stor.indexString(cnt), mode=FILE_READ) as loadResidual:
                    rho = Function(V, name='Residual')
                    loadResidual.load(rho)
                    loadResidual.close()

                # Estimate error using forward residual
                epsilon = assemble(v * rho * dual * dx)
                norm = assemble(epsilon * dx)
                epsilon.dat.data[:] = np.abs(epsilon.dat.data) / norm
                epsilon.rename("Error indicator")

                # Save error indicator data to HDF5
                if not cnt % rm:
                    with DumbCheckpoint(dirName + 'hdf5/error_' + stor.indexString(cnt), mode=FILE_CREATE) as saveError:
                        saveError.store(epsilon)
                        saveError.close()

                # Print to screen, save data and increment counters
                print('t = %.3fs' % t)
                adjointFile.write(dual, time=t)
                errorFile.write(epsilon, time=t)
                t -= dt
                cnt -= 1
                save = False
            else:
                save = True
        except:
            continue
    dualTimer = clock() - dualTimer
    print('Adjoint run complete. Run time: %.3fs' % dualTimer)
    t += dt
    cnt += 1

    # Reset initial conditions for primal problem and recreate error indicator placeholder
    phi = ic.copy(deepcopy=True)
    phi.rename('Concentration')
    epsilon = Function(P0, name="Error indicator")

print('Starting adaptive mesh primal run (forwards in time)')
adaptTimer = clock()
while t <= T:
    if not cnt % rm:
        stepTimer = clock()

        # Reconstruct Hessian
        W = TensorFunctionSpace(mesh, "CG", 1)
        H = adap.constructHessian(mesh, W, phi, op=op)

        # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
        if useAdjoint:
            with DumbCheckpoint(dirName + 'hdf5/error_' + stor.indexString(cnt), mode=FILE_READ) as loadError:
                loadError.load(epsilon)  # P0 field on the initial mesh
                loadError.close()
            errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
            for k in range(mesh.topology.num_vertices()):
                H.dat.data[k] *= errEst.dat.data[k]  # Scale by error estimate

        # Adapt mesh and interpolate variables
        M = adap.computeSteadyMetric(mesh, W, H, phi, nVerT=nVerT, op=op)
        if op.gradate:
            adap.metricGradation(mesh, M)
        if op.advect:
            M = adap.advectMetric(M, w, dt, n=rm, nu=nu)
        mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
        phi = inte.interp(mesh, phi)[0]
        phi.rename("Concentration")
        V = FunctionSpace(mesh, "CG", 2)
        phi_next = Function(V, name="Concentration next")

        # Re-establish bilinear form and set boundary conditions
        psi = TestFunction(V)
        w = Function(VectorFunctionSpace(mesh, "CG", 2), name='Wind field').interpolate(Expression([1, 0]))
        F = form.weakResidualAD(phi_next, phi, psi, w, Dt, nu=nu)
        bc = DirichletBC(V, 0., "on_boundary")

        # Get mesh stats
        nEle = msh.meshStats(mesh)[0]
        N = [min(nEle, N[0]), max(nEle, N[1])]
        Sn += nEle
        op.printToScreen(cnt / rm + 1, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, N)

    # Solve problem at current timestep
    solve(F == 0, phi_next, bc)
    phi.assign(phi_next)

    # Print to screen, save data and increment counters
    print('t = %.3fs' % t)
    adaptiveFile.write(phi, time=t)
    t += dt
    cnt += 1
adaptTimer = clock() - adaptTimer
print('Adaptive primal run complete. Run time: %.3fs' % adaptTimer)

# Print to screen timing analyses
if useAdjoint:
    print("""******** TIMINGS ********
Forward run   %5.3fs
Adjoint run   %5.3fs
Adaptive run  %5.3fs""" % (primalTimer, dualTimer, adaptTimer))
