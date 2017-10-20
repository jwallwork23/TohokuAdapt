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


print('******************************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ********************************\n')
print('Mesh adaptive solver initially defined on a mesh of',)

# Define initial mesh:
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 5): ') or 5))
N1 = len(mesh.coordinates.dat.data)     # Minimum number of vertices
N2 = N1                                 # Maximum number of vertices
SumN = N1                               # Sum over vertex counts
print('...... mesh loaded. Initial number of vertices : ', N1)
gauge = input('Gauge choice from {P02, P06, 801, 802 803, 804, 806}? (default P02): ') or 'P02'

# Get default adaptivity parameter values:
op = opt.Options(gauge=gauge)
numVer = op.vscale * N1
hmin = op.hmin
hmax = op.hmax
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = op.ntype
mtype = op.mtype
iso = op.iso
if not iso:
    hess_meth = op.hessMeth

# Get physical parameters:
g = op.g

# Get Courant number adjusted timestepping parameters:
T = op.T
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

# Get gauge coordinates:
gCoord = op.gaugeCoord()

# Initialise counters, files and gauge data measurements:
t = 0.
dumpn = 0
mn = 0
filename = 'plots/simpleAdapt/'
if not iso:
    filename += 'an'
filename += 'isotropic'

print('\nEntering outer timeloop!')
tic1 = clock()
while t < T - 0.5 * dt:
    mn += 1
    tic2 = clock()

    ### GET / LOAD VARIABLES

    # Compute Hessian and metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if iso:
        M = Function(V)
        if mtype == 's':
            raise NotImplementedError('Adaption w.r.t. speed not yet implemented.')
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
            raise NotImplementedError('Adaption w.r.t. speed not yet implemented.')
        if mtype != 's':
            H = adap.constructHessian(mesh, V, eta, method=hessMeth)
            M2 = adap.computeSteadyMetric(mesh, V, H, eta, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
        if mtype == 'b':
            raise NotImplementedError('Adaption w.r.t. two fields not yet implemented.')
        else:
            M = Function(V)
            M.assign(M2)

    # Adapt mesh with respect to computed metric field and interpolate functions onto new mesh:
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh

    ### INTERPOLATE

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += n
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n


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
