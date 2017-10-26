from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import math
import sys

import utils.adaptivity as adap
import utils.domain as dom
import utils.interpolation as inte
import utils.options as opt
import utils.storage as stor

print('************** ADJOINT-BASED ADAPTIVE TSUNAMI SIMULATION **************\n')

# Define initial mesh:
print('ADJOINT-GUIDED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 4): ') or 4))
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
mesh0 = mesh
W0 = VectorFunctionSpace(mesh0, 'DG', 1) * FunctionSpace(mesh0, 'DG', 1)
print('...... mesh loaded. Initial #Vertices : %d. Initial #Elements : %d. \n' % (nVer, nEle))
gauge = input('Gauge choice from {P02, P06, 801, 802 803, 804, 806}? (default P02): ') or 'P02'

# Get default adaptivity parameter values:
op = opt.Options(vscale=0.2, mtype='f', rm=60, gauge=gauge)
numVer = op.vscale * nVer
hmin = op.hmin
hmax = op.hmax
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = op.ntype
mtype = op.mtype
beta = op.beta
iso = op.iso
if not iso:
    hessMeth = op.hessMeth

# Get physical parameters:
g = op.g

# Get solver parameters:
T = op.T
Ts = op.Ts
dt = op.dt
Dt = Constant(dt)
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print('WARNING: chosen timestep dt = %.2fs exceeds recommended value of %.2fs' % (dt, cdt))
    if bool(input('Hit anything except enter if happy to proceed.')) or False:
        exit(23)
else:
    print('Using Courant number adjusted timestep dt = %.2fs' % dt)
ndump = op.ndump
rm = op.rm
stored = bool(input('Hit anything but enter if adjoint data is already stored: ')) or False

tic1 = clock()

if not stored:
    # Initalise counters:
    t = T
    i = -1
    dumpn = ndump
    meshn = rm

    # Forcing switch:
    coeff = Constant(1.)
    switch = True

    # TODO: how to consider the adjoint equations in Thetis?...

    # Establish adjoint variables and apply initial conditions:
    lam_ = Function(W0)
    lu_, le_ = lam_.split()
    lu_.interpolate(Expression([0, 0]))
    le_.interpolate(Expression(0))

    # Establish smoothened indicator function for adjoint equations:
    f = Function(W0.sub(1), name='Forcing term')
    f.interpolate(Expression('(x[0] > 490e3) & (x[0] < 640e3) & (x[1] > 4160e3) & (x[1] < 4360e3) ? ' +
                             'exp(1. / (pow(x[0] - 565e3, 2) - pow(75e3, 2))) * ' +
                             'exp(1. / (pow(x[1] - 4260e3, 2) - pow(100e3, 2))) : 0.'))

    # Set up dependent variables of the adjoint problem:
    lam = Function(W0)
    lam.assign(lam_)
    lu, le = lam.split()
    lu.rename('Adjoint velocity')
    le.rename('Adjoint free surface')

    # Store final time data to HDF5 and PVD:
    with DumbCheckpoint('data/adjointSolution_{y}'.format(y=i), mode=FILE_CREATE) as chk:
        chk.store(lu)
        chk.store(le)
        chk.close()
    adjointFile = File('plots/adjointBased/adjoint.pvd')
    adjointFile.write(lu, le, time=T)

    # Establish test functions and midpoint averages:
    w, xi = TestFunctions(W)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Set up the variational problem:
    La = ((le - le_) * xi - Dt * g * inner(luh, grad(xi)) - coeff * f * xi
          + inner(lu - lu_, w) + Dt * (b * inner(grad(leh), w) + leh * inner(grad(b), w))) * dx
    adjointProblem = NonlinearVariationalProblem(La, lam)
    adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=op.params)

    # Split to access data:
    lu, le = lam.split()
    lu_, le_ = lam_.split()

    print('\nStarting fixed resolution adjoint run...')
    tic2 = clock()
    while t > 0.5 * dt:

        # Increment counters:
        t -= dt
        dumpn -= 1
        meshn -= 1

        # Modify forcing term:
        if (t < Ts + 1.5 * dt) & switch:
            coeff.assign(0.5)
        elif (t < Ts + 0.5 * dt) & switch:
            switch = False
            coeff.assign(0.)

    # Solve the problem and update:
        adjointSolver.solve()
        lam_.assign(lam)

        # Dump to vtu:
        if dumpn == 0:
            dumpn += ndump
            adjointFile.write(lu, le, time=t)

        # Dump to HDF5:
        if meshn == 0:
            meshn += rm
            i -= 1
            # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
            if not stored:
                print('t = %1.1fs' % t)
                with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
                    chk.store(lu)
                    chk.store(le)
                    chk.close()

    toc2 = clock()
    print('... done! Elapsed time for adjoint solver: %1.2fs' % (toc2 - tic2))

# Initialise counters:
dumpn = 0   # Dump counter
mn = 0      # Mesh number
Sn = 0      # Sum over #Elements

# Approximate isotropic metric at boundaries of initial mesh using circumradius:
h = Function(W.sub(1))
h.interpolate(CellSize(mesh0))
M_ = Function(TensorFunctionSpace(mesh0, 'CG', 1))
for j in DirichletBC(W.sub(1), 0, 'on_boundary').nodes:
    h2 = pow(h.dat.data[j], 2)
    M_.dat.data[j][0, 0] = 1. / h2
    M_.dat.data[j][1, 1] = 1. / h2

print('\nStarting mesh adaptive forward run...')
while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Define discontinuous spaces on the new mesh:
    elev_2d = Function(FunctionSpace(mesh, 'DG', 1))
    uv_2d = Function(VectorFunctionSpace(mesh, 'DG', 1))

    # Enforce initial conditions on discontinuous space / load variables from disk:
    index = mn * int(rm / ndump)
    indexStr = stor.indexString(index)
    dirName = 'plots/adjointBased/'

    if mn == 0:
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(Expression((0, 0)))
    else:
        with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as el:
            el.load(elev_2d, name='elev_2d')
            el.close()
        with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as ve:
            ve.load(uv_2d, name='uv_2d')
            ve.close()

    # Create functions to hold inner product and significance data:
    DG1 = FunctionSpace(mesh, 'DG', 1)
    ip = Function(DG1, name='Inner product')
    significance = Function(DG1, name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        # Read in saved data from HDF5:
        with DumbCheckpoint('data/adjointSolution_{y}'.format(y=i), mode=FILE_READ) as chk:
            lu = Function(W0.sub(0), name='Adjoint velocity')
            le = Function(W0.sub(1), name='Adjoint free surface')
            chk.load(lu)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if mn != 1:
            print('    #### Interpolation step %d / %d'
                  % (j - max(i, int((Ts - T) / (dt * ndump))) + 1, len(range(max(i, int((Ts - T) / (dt * ndump))), 0))))
            lu, le = interp(mesh, lu, le)

        # Multiply fields together:
        ip.dat.data[:] = lu.dat.data[:, 0] * uv_2d.dat.data[:, 0] + lu.dat.data[:, 1] * uv_2d.dat.data[:, 1]
        ip.dat.data[:] += le.dat.data * elev_2d.dat.data

        # Extract (pointwise) maximal values:
        if j == 0:
            significance.dat.data[:] = ip.dat.data[:]
        else:
            for k in range(len(ip.dat.data)):
                if np.abs(ip.dat.data[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = ip.dat.data[k]
    significanceFile.write(significance, time=t)

    # Interpolate initial mesh size onto new mesh and build associated metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    fields = inte.interp(mesh, h)
    h = Function(W.sub(1))
    h.dat.data[:] = fields[0].dat.data[:]
    M_ = Function(V)
    for j in DirichletBC(W.sub(1), 0, 'on_boundary').nodes:
        h2 = pow(h.dat.data[j], 2)
        M_.dat.data[j][0, 0] = 1. / h2
        M_.dat.data[j][1, 1] = 1. / h2

    # Generate metric associated with significant data:
    if iso:
        M = Function(V)
        for j in range(len(M.dat.data)):
            isig2 = 1. / max(hmin2, min(pow(significance.dat.data[j], 2), hmax2))
            M.dat.data[j][0, 0] = isig2
            M.dat.data[j][1, 1] = isig2
    else:
        H = Function(V)
        if mtype == 's':
            spd = Function(W.sub(1))
            spd.interpolate(sqrt(dot(u, u)))
            H = adap.constructHessian(mesh, V, spd, method=hessMeth)
        elif mtype == 'f':
            H = adap.constructHessian(mesh, V, eta, method=hessMeth)
        else:
            raise NotImplementedError('Cannot currently perform goal-based adaption with respect to two fields.')
        for k in range(mesh.topology.num_vertices()):
            H.dat.data[k] *= significance.dat.data[k]
        M = adap.computeSteadyMetric(mesh, V, H, eta, h_min=hmin, h_max=hmax, normalise=ntype, num=numVer)

    # Gradate metric, adapt mesh and interpolate variables:
    M = adap.metricIntersection(mesh, V, M, M_, bdy=True)
    adap.metricGradation(mesh, M, beta, isotropic=iso)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = inte.interpTaylorHood(mesh, u, u_, eta, eta_, b)
    i += 1

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += n
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n

    # Establish test functions and midpoint averages:
    v, ze = TestFunctions(W)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem:
    Lf = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    forwardProblem = NonlinearVariationalProblem(Lf, q)
    forwardSolver = NonlinearVariationalSolver(forwardProblem, solver_parameters=op.params)

    # Split to access data and relabel functions:
    u, eta = q.split()
    u_, eta_ = q_.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Inner timeloop:
    for k in range(rm):
        t += dt
        dumpn += 1

        # Solve the problem and update:
        forwardSolver.solve()
        q_.assign(q)

        # Store data:
        gaugeData.append(eta.at(gcoord))
        if dumpn == ndump:
            dumpn -= ndump
            forwardFile.write(u, eta, time=t)
    toc2 = clock()

    mn += 1
    print('\n************ Adaption step %d **************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (t / 60., T / 60.))
    print('Number of vertices after adaption step %d: ' % mn, n)
    print('Min/max vertex counts: %d, %d' % (N1, N2))
    print('Mean vertex count: %d' % (float(SumN) / mn))
    print('Elapsed time for this step: %1.2fs' % (toc2 - tic2), '\n')
print('\a')
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
