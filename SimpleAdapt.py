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


print('******************************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ********************************\n')
print('Mesh adaptive solver initially defined on a mesh of',)

# Define initial mesh:
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 5): ') or 5))
N1 = len(mesh.coordinates.dat.data)     # Minimum number of vertices
N2 = N1                                 # Maximum number of vertices
SumN = N1                               # Sum over vertex counts
print('...... mesh loaded. Initial number of vertices : ', N1, 'More options...')

# Set physical parameters:
g = 9.81                        # Gravitational acceleration (m s^{-2})

numVer = float(input('Target vertex count as a proportion of the initial number? (default 0.85): ') or 0.85) * N1
hmin = float(input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
hmax = float(input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = input('Normalisation type? (lp/manual, default lp): ') or 'lp'
mtype = input('Adapt with respect to speed, free surface or both? (s/f/b, default b): ') or 'b'
if mtype not in ('s', 'f', 'b'):
    raise ValueError('Field selection not recognised. lease try again, choosing s, f or b.')
mat_out = bool(input('Hit any key to output Hessian and metric: ')) or False
iso = bool(input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
if not iso:
    hess_meth = input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'

# Courant number adjusted timestepping parameters:
T = float(input('Simulation duration in minutes (default 25)?: ') or 25.) * 60.
dt = float(input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print ('WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt)
    if bool(input('Hit anything except enter if happy to proceed.')) or False:
        exit(23)
ndump = int(15. / dt)           # Timesteps per data dump
rm = int(input('Timesteps per re-mesh (default 30)?: ') or 30)

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

# Establish mixed function space and initial conditions:
W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
q_ = Function(W)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.assign(eta0)

# Set up functions of the weak problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Initialise counters, files and gauge data measurements:
t = 0.
dumpn = 0
mn = 0
u.rename('Fluid velocity')
eta.rename('Free surface displacement')
if iso:
    q_file = File('plots/isotropic_outputs/tsunami.pvd')
    if mat_out:
        m_file = File('plots/isotropic_outputs/tsunami_metric.pvd')
        h_file = File('plots/isotropic_outputs/tsunami_hessian.pvd')
else:
    q_file = File('plots/anisotropic_outputs/tsunami.pvd')
    if mat_out:
        m_file = File('plots/anisotropic_outputs/tsunami_metric.pvd')
        h_file = File('plots/anisotropic_outputs/tsunami_hessian.pvd')
q_file.write(u, eta, time=0)
gauge_dat = [eta.at(gcoord)]
print('\nEntering outer timeloop!')
tic1 = clock()
while t < T - 0.5 * dt:
    mn += 1
    tic2 = clock()

    # Compute Hessian and metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if iso:
        M = Function(V)
        if mtype == 's':
            spd2 = Function(FunctionSpace(mesh, 'CG', 1))
            spd2.interpolate(dot(u, u))
            for i in range(len(M.dat.data)):
                ispd2 = 1. / max(hmin2, min(spd2.dat.data[i], hmax2))
                M.dat.data[i][0, 0] = ispd2
                M.dat.data[i][1, 1] = ispd2
        elif mtype == 'f':
            for i in range(len(M.dat.data)):
                ieta2 = 1. / max(hmin2, min(pow(eta.dat.data[i], 2), hmax2))
                M.dat.data[i][0, 0] = ieta2
                M.dat.data[i][1, 1] = ieta2
        else:
            raise NotImplementedError('Cannot currently interpret isotropic adaption with respect to two fields.')
    else:
        H = Function(V)
        if mtype != 'f':
            spd = Function(W.sub(1))
            spd.interpolate(sqrt(dot(u, u)))
            H = adap.constructHessian(mesh, V, spd, method=hess_meth)
            M = adap.computeSteadyMetric(mesh, V, H, spd, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
        if mtype != 's':
            H = adap.constructHessian(mesh, V, eta, method=hess_meth)
            M2 = adap.computeSteadyMetric(mesh, V, H, eta, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
        if mtype == 'b':
            M = adap.metricIntersection(mesh, V, M, M2)
        else:
            M = Function(V)
            M.assign(M2)

    # Adapt mesh with respect to computed metric field and interpolate functions onto new mesh:
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = inte.interpTaylorHood(mesh, u, u_, eta, eta_, b)

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += n
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n

    # Set up functions of weak problem:
    v, ze = TestFunctions(W)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem
    L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters={'mat_type': 'matfree',
                                                                   'snes_type': 'ksponly',
                                                                   'pc_type': 'python',
                                                                   'pc_python_type': 'firedrake.AssembledPC',
                                                                   'assembled_pc_type': 'lu',
                                                                   'snes_lag_preconditioner': -1,
                                                                   'snes_lag_preconditioner_persists': True})
    # Split to access data and relabel functions:
    u_, eta_ = q_.split()
    u, eta = q.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Inner timeloop:
    for j in range(rm):
        t += dt
        dumpn += 1
        q_solv.solve()  # Solve problem
        q_.assign(q)    # Update variables

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

    # Print to screen:
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