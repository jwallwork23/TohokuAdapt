from thetis import *
from thetis.field_defs import field_metadata

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

tic1 = clock()
while t < T - 0.5 * dt:
    tic2 = clock()

    index = mn * int(rm / ndump)
    if index in range(1, 5):
        indexStr = '0000' + str(index)
    elif index in range(5, 50):
        indexStr = '000' + str(index)
    elif index in range(50, 500):
        indexStr = '00' + str(index)
    elif index in range(500, 5000):
        indexStr = '0' + str(index)
    dirName = 'plots/simpleAdapt/'

    # Define discontinuous spaces on the new mesh:
    elev_2d = Function(FunctionSpace(mesh, 'DG', 1))
    uv_2d = Function(VectorFunctionSpace(mesh, 'DG', 1))
    if mn == 0:
        # Enforce initial conditions on discontinuous space:
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(Expression((0, 0)))
    else:
        # Load variables from disk:
        # print('#### DEBUG: Attempting to load ', dirName + 'hdf5/Elevation2d_' + indexStr)
        with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as el:
            el.load(elev_2d, name='elev_2d')
        with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as ve:
            ve.load(uv_2d, name='uv_2d')

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

    # Get solver parameter values and construct solver:     TODO: different FE space options?
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.simulation_export_time = dt * ndump
    options.simulation_end_time = (mn + 1) * dt * rm
    options.timestepper_type = 'CrankNicolson'
    options.timestep = dt

    # Specify outfile directory and HDF5 checkpointing:
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

    # Load state:
    field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
    e = exporter.ExportManager(dirName + 'hdf5',
                               ['elev_2d', 'uv_2d'],
                               field_dict,
                               field_metadata,
                               export_type='hdf5')
    solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)

    # Timestepper bookkeeping for export time step
    solver_obj.i_export = index * mn
    solver_obj.next_export_t = (mn + 1) * dt * ndump
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t / dt))
    solver_obj.simulation_time = mn * rm

    # For next export
    solver_obj.export_initial_state = (dirName != options.output_directory)
    if solver_obj.export_initial_state:
        offset = 0
    else:
        offset = 1
    solver_obj.next_export_t += options.simulation_export_time
    for e in solver_obj.exporters.values():
        e.set_next_export_ix(solver_obj.i_export + offset)

    # Time integrate
    solver_obj.iterate()
    toc2 = clock()

    # Print to screen:
    mn += 1
    print('\n************************ Adaption step %d **************************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (mn * dt * ndump / 60., T / 60.))
    print('#Vertices after adaption step %d: %d' % (mn, n))
    print('Min/max #Vertices: %d. Mean #Vertices: %d' % (N, float(Sn) / mn))
    print('Elapsed time for this step: %1.2fs \n' % (toc2 - tic2))
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
