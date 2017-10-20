from thetis import *
import numpy as np
from time import clock
import math
import sys

import utils.adaptivity as adap
import utils.domain as dom
import utils.interpolation as inte
import utils.options as opt


print('******************************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ********************************\n')

# Define initial mesh:
print('Mesh adaptive solver initially defined on a mesh of',)
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 5): ') or 5))
N = [len(mesh.coordinates.dat.data), len(mesh.coordinates.dat.data)]    # Min/max number of vertices
Sn = N[0]   # Sum over #Vertices
print('...... mesh loaded. Initial number of vertices : ', N[0])

# Get default adaptivity parameter values:
op = opt.Options()
numVer = op.vscale * N[0]
hmin = op.hmin
hmax = op.hmax
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = op.ntype
mtype = op.mtype
iso = op.iso
if not iso:
    hessMeth = op.hessMeth

# Get physical parameters:
g = op.g

# Get Courant number adjusted timestepping parameters:
T = op.T
dt = op.dt
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print('WARNING: chosen timestep dt = %.2fs exceeds recommended value of %.2fs' % (dt, cdt))
    if bool(input('Hit anything except enter if happy to proceed.')) or False:
        exit(23)
ndump = op.ndump
rm = op.rm

# Initialise counters and output directory:
t = 0.
dumpn = 0
mn = 0
index = int(rm / ndump)
if index in range(1, 5):
    index = '0000' + str(index)
elif index in range(5, 50):
    index = '000' + str(index)
elif index in range(50, 500):
    index = '00' + str(index)
elif index in range(500, 5000):
    index = '0' + str(index)
dirName = 'plots/simpleAdapt/'

tic1 = clock()
while t < T - 0.5 * dt:
    mn += 1
    tic2 = clock()

    # Define discontinuous spaces on the new mesh:
    elev_2d = Function(FunctionSpace(mesh, 'DG', 1))
    uv_2d = Function(VectorFunctionSpace(mesh, 'DG', 1))
    if mn == 1:
        # Enforce initial conditions on discontinuous space:
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(Expression((0, 0)))
    else:
        # Load variables from disk:
        # print('#### DEBUG: Attempting to load ', dirName + 'hdf5/Elevation2d_' + index)
        with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + index, mode=FILE_READ) as el:
            el.load(elev_2d, name='elev_2d')
        with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + index, mode=FILE_READ) as uv:
            uv.load(uv_2d, name='uv_2d')

    # Compute Hessian and metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if iso:
        M = Function(V)
        for i in range(len(M.dat.data)):
            ielev2 = 1. / max(hmin2, min(pow(elev_2d.dat.data[i], 2), hmax2))
            M.dat.data[i][0, 0] = ieta2
            M.dat.data[i][1, 1] = ieta2
    else:
        H = adap.constructHessian(mesh, V, elev_2d, method=hessMeth)
        M = adap.computeSteadyMetric(mesh, V, H, elev_2d, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)

    # Adapt mesh with respect to computed metric field and interpolate functions onto new mesh:
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    Sn += n
    N = [min(n, N[0]), max(n, N[1])]

    # Get solver parameter values and construct solver:     TODO: different FE space options
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.simulation_export_time = dt * ndump
    options.simulation_end_time = dt * rm
    options.timestepper_type = 'CrankNicolson'
    options.timestep = dt

    # Specify outfile directory and HDF5 checkpointing:
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

    # Apply initial conditions and time integrate:
    solver_obj.assign_initial_conditions(elev=elev_2d)
    solver_obj.iterate()
    toc2 = clock()

    # Print to screen:
    print('\n************ Adaption step %d **************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (t / 60., T / 60.))
    print('#Vertices after adaption step %d: %d' % (mn, n))
    print('Min/max #Vertices:', N)
    print('Mean #Vertices: %d' % (float(Sn) / mn))
    print('Elapsed time for this step: %1.2fs \n' % (toc2 - tic2))
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
