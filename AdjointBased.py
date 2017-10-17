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
print('...... mesh loaded. Initial number of vertices : ', N1, '\nMore options...')

# Set up adaptivity parameters:
numVer = float(input('Target vertex count as a proportion of the initial number? (default 0.2): ') or 0.2) * N1
hmin = float(input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
hmax = float(input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = input('Normalisation type? (lp/manual): ') or 'lp'
mat_out = bool(input('Hit anything but enter to output Hessian and metric: ')) or False
beta = float(input('Metric gradation scaling parameter (default 1.4): ') or 1.4)
iso = bool(input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
if not iso:
    hess_meth = input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'
    mtype = input('Adapt with respect to speed, free surface or both? (s/f/b, default f): ') or 'f'
    if mtype not in ('s', 'f', 'b'):
        raise ValueError('Field selection not recognised. Please try again, choosing s, f or b.')

# Specify parameters:
T = float(input('Simulation duration in minutes (default 25)?: ') or 25.) * 60.
Ts = 5. * 60.                   # Time range lower limit (s), during which we can assume the wave won't reach the shore
g = 9.81                        # Gravitational acceleration (m s^{-2})
dt = float(input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print('WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt)
    if bool(input('Hit anything except enter if happy to proceed.')) or False:
        exit(23)
ndump = int(15. / dt)           # Timesteps per data dump
rm = int(input('Timesteps per re-mesh (default 60)?: ') or 60)
stored = bool(input('Hit anything but enter if adjoint data is already stored: ')) or False

# Convert gauge locations to UTM coordinates:
glatlon = {'P02': (38.5002, 142.5016), 'P06': (38.6340, 142.5838),
           '801': (38.2, 141.7), '802': (39.3, 142.1), '803': (38.9, 141.8), '804': (39.7, 142.2), '806': (37.0, 141.2)}

gloc = {}
for key in glatlon:
    east, north, zn, zl = conv.from_latlon(glatlon[key][0], glatlon[key][1], force_zone_number=54)
    gloc[key] = (east, north)

# Set gauge arrays:
gtype = input('Pressure or tide gauge? (p/t, default p): ') or 'p'
if gtype == 'p':
    gauge = input('Gauge P02 or P06? (default P02): ') or 'P02'
    gcoord = gloc[gauge]
elif gtype == 't':
    gauge = input('Gauge 801, 802, 803, 804 or 806? (default 801): ') or '801'
    gcoord = gloc[gauge]
else:
    ValueError('Gauge type not recognised. Please choose p or t.')

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

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
    with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
        chk.store(lu_P1)
        chk.store(le)
    lam_file = File('plots/goal-based_outputs/tsunami_adjoint.pvd')
    lam_file.write(lu, le, time=T)

    # Establish test functions and midpoint averages:
    w, xi = TestFunctions(W)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Set up the variational problem:
    La = ((le - le_) * xi - Dt * g * inner(luh, grad(xi)) - coeff * f * xi
          + inner(lu - lu_, w) + Dt * (b * inner(grad(leh), w) + leh * inner(grad(b), w))) * dx
    lam_prob = NonlinearVariationalProblem(La, lam)
    lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

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
        lam_solv.solve()
        lam_.assign(lam)

        # Dump to vtu:
        if dumpn == 0:
            dumpn += ndump
            lam_file.write(lu, le, time=t)

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
    q_file = File('plots/goal-based_outputs/tsunami_forward_iso.pvd')
    sig_file = File('plots/goal-based_outputs/tsunami_significance_iso.pvd')
    if mat_out:
        m_file = File('plots/goal-based_outputs/tsunami_metric_iso.pvd')
        h_file = File('plots/goal-based_outputs/tsunami_hessian_iso.pvd')
else:
    q_file = File('plots/goal-based_outputs/tsunami_forward.pvd')
    sig_file = File('plots/goal-based_outputs/tsunami_significance.pvd')
    if mat_out:
        m_file = File('plots/goal-based_outputs/tsunami_metric.pvd')
        h_file = File('plots/goal-based_outputs/tsunami_hessian.pvd')
q_file.write(u, eta, time=0)
gauge_dat = [eta.at(gcoord)]

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
        with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_READ) as chk:
            lu_P1 = Function(W0.sub(0), name='P1 adjoint velocity')
            le = Function(W0.sub(1), name='Adjoint free surface')
            chk.load(lu_P1)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if mn != 1:
            print('    #### Interpolation step', j - max(i, int((Ts - T) / (dt * ndump))) + 1, '/', \
                len(range(max(i, int((Ts - T) / (dt * ndump))), 0)))
            lu_P1, le = interp(mesh, lu_P1, le)

        # Multiply fields together:
        ip.dat.data[:] = lu_P1.dat.data[:, 0] * vel.dat.data[:, 0] + lu_P1.dat.data[:, 1] * vel.dat.data[:, 1] \
                         + le.dat.data * eta.dat.data

        # Extract (pointwise) maximal values:
        if j == 0:
            significance.dat.data[:] = ip.dat.data[:]
        else:
            for k in range(len(ip.dat.data)):
                if np.abs(ip.dat.data[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = ip.dat.data[k]
    sig_file.write(significance, time=t)

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
            H = adap.constructHessian(mesh, V, spd, method=hess_meth)
        elif mtype == 'f':
            H = adap.constructHessian(mesh, V, eta, method=hess_meth)
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
    q_prob = NonlinearVariationalProblem(Lf, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

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
        q_solv.solve()
        q_.assign(q)

        # Store data:
        gauge_dat.append(eta.at(gcoord))
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, eta, time=t)
            if mat_out:
                H.rename('Hessian')
                M.rename('Metric')
                h_file.write(H, time=t)
                m_file.write(M, time=t)
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
stor.gauge_timeseries(gauge, gauge_dat)