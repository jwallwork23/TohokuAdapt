from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from time import clock
import math
import sys

import utils.adaptivity as adap
import utils.conversion as conv
import utils.domain as dom
import utils.interpolation as inte
import utils.options as opt
import utils.storage as stor


print('\n******************************** GOAL-BASED ADAPTIVE TSUNAMI SIMULATION ********************************\n')
print('GOAL-BASED, mesh adaptive solver initially defined on a mesh of')
tic1 = clock()

# Define initial mesh:
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 4): ') or 4))
mesh0 = mesh
W0 = VectorFunctionSpace(mesh0, 'CG', 1) * FunctionSpace(mesh0, 'CG', 1)    # P1-P1 space for interpolating velocity
N1 = len(mesh.coordinates.dat.data)                                         # Minimum number of vertices
N2 = N1                                                                     # Maximum number of vertices
SumN = N1                                                                   # Sum over vertex counts
print('...... mesh loaded. Initial number of vertices : ', N1)
gauge = input('Gauge choice from {P02, P06, 801, 802 803, 804, 806}? (default P02): ') or 'P02'

# Get default adaptivity parameter values:
op = opt.Options(vscale=0.2, mtype='f', rm=60, gauge=gauge)
numVer = op.vscale * N1
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

# Get gauge coordinates:
gCoord = op.gaugeCoord()

# Initalise counters:
t = T
i = -1
dumpn = ndump
meshn = rm

# Forcing switch:
coeff = Constant(1.)
switch = True

# Establish mixed function spaces and associated initial conditions:
W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
q_ = Function(W)
lam_ = Function(W)
u_, eta_ = q_.split()
lu_, le_ = lam_.split()
u_.interpolate(Expression([0, 0]))
eta_.assign(eta0)
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression(0))

if not stored:
    # Establish smoothened indicator function for adjoint equations:
    f = Function(W.sub(1), name='Forcing term')
    f.interpolate(Expression('(x[0] > 490e3) & (x[0] < 640e3) & (x[1] > 4160e3) & (x[1] < 4360e3) ? ' +
                             'exp(1. / (pow(x[0] - 565e3, 2) - pow(75e3, 2))) * ' +
                             'exp(1. / (pow(x[1] - 4260e3, 2) - pow(100e3, 2))) : 0.'))

    # Set up dependent variables of the adjoint problem:
    lam = Function(W)
    lam.assign(lam_)
    lu, le = lam.split()
    lu.rename('Adjoint velocity')
    le.rename('Adjoint free surface')

    # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
    lu_P1 = Function(VectorFunctionSpace(mesh, 'CG', 1), name='P1 adjoint velocity')
    lu_P1.interpolate(lu)
    with DumbCheckpoint('data/adjointSolution_{y}'.format(y=i), mode=FILE_CREATE) as chk:
        chk.store(lu_P1)
        chk.store(le)
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
    if not stored:
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
                lu_P1.interpolate(lu)
                with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
                    chk.store(lu_P1)
                    chk.store(le)
if not stored:
    toc2 = clock()
    print('... done! Elapsed time for adjoint solver: %1.2fs' % (toc2 - tic2))

# Set up dependent variables of the forward problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Label variables:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Intialise files:
if iso:
    forwardFile = File('plots/adjointBased/isotropicForward.pvd')
    significanceFile = File('plots/adjointBased/isotropicSignificance.pvd')
else:
    forwardFile = File('plots/adjointBased/anisotropicForward.pvd')
    significanceFile = File('plots/adjointBased/anisotropicSignificance.pvd')
forwardFile.write(u, eta, time=0)
gaugeData = [eta.at(gcoord)]

# Initialise counters:
t = 0.
dumpn = 0
mn = 0

# Approximate isotropic metric at boundaries of initial mesh using circumradius:
h = Function(W.sub(1))
h.interpolate(CellSize(mesh0))
M_ = Function(TensorFunctionSpace(mesh0, 'CG', 1))
for j in DirichletBC(W.sub(1), 0, 'on_boundary').nodes:
    h2 = pow(h.dat.data[j], 2)
    M_.dat.data[j][0, 0] = 1. / h2
    M_.dat.data[j][1, 1] = 1. / h2

print('\nStarting mesh adaptive forward run...')
while t < T - 0.5 * dt:
    mn += 1
    tic2 = clock()

    # Compute Hessian:
    V = TensorFunctionSpace(mesh, 'CG', 1)

    # Interpolate velocity in a P1 space:
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    vel.interpolate(u)

    # Create functions to hold inner product and significance data:
    ip = Function(W.sub(1), name='Inner product')
    significance = Function(W.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        # Read in saved data from .h5:
        with DumbCheckpoint('data/adjointSolution_{y}'.format(y=i), mode=FILE_READ) as chk:
            lu_P1 = Function(W0.sub(0), name='P1 adjoint velocity')
            le = Function(W0.sub(1), name='Adjoint free surface')
            chk.load(lu_P1)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if mn != 1:
            print('    #### Interpolation step %d / %d'
                  % (j - max(i, int((Ts - T) / (dt * ndump))) + 1, len(range(max(i, int((Ts - T) / (dt * ndump))), 0))))
            lu_P1, le = interp(mesh, lu_P1, le)

        # Multiply fields together:
        ip.dat.data[:] = lu_P1.dat.data[:, 0] * vel.dat.data[:, 0] + lu_P1.dat.data[:, 1] * vel.dat.data[:, 1]
        ip.dat.data[:] += le.dat.data * eta.dat.data

        # Extract (pointwise) maximal values:
        if j == 0:
            significance.dat.data[:] = ip.dat.data[:]
        else:
            for k in range(len(ip.dat.data)):
                if np.abs(ip.dat.data[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = ip.dat.data[k]
    significanceFile.write(significance, time=t)

    # Interpolate initial mesh size onto new mesh and build associated metric:
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

    print('\n************ Adaption step %d **************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (t / 60., T / 60.))
    print('Number of vertices after adaption step %d: ' % mn, n)
    print('Min/max vertex counts: %d, %d' % (N1, N2))
    print('Mean vertex count: %d' % (float(SumN) / mn))
    print('Elapsed time for this step: %1.2fs' % (toc2 - tic2), '\n')
print('\a')
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))

# Store gauge timeseries data to file:
stor.gaugeTimeseries(gauge, gaugeData)
