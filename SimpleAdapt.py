from thetis import *
from thetis.field_defs import field_metadata

import numpy as np
from time import clock
import math

import utils.adaptivity as adap
import utils.domain as dom
import utils.interpolation as inte
import utils.options as opt
import utils.storage as stor


print('*************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ***************\n')

# Define initial mesh and mesh statistics placeholders:
print('MESH ADAPTIVE solver initially defined on a mesh of')
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 5): ') or 5))
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('...... mesh loaded. Initial #Vertices : %d. Initial #Elements : %d. \n' % (nVer, nEle))
# TODO: build in functionality for loading a previous state. NOTE this may involve saving the mesh to disk...
# resume = input('Hit anything except enter to resume a previous simulation.')
# if resume:
#     iStart = int(input('Simulation starting index (default i = 0)?: ')) or 0

# Get default adaptivity parameter values:
op = opt.Options()
numVer = op.vscale * nVer
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

# Initialise counters and filenames:
dumpn = 0   # Dump counter
mn = 0      # Mesh number
Sn = 0      # Sum over #Elements
dirName = 'plots/simpleAdapt/'
if iso:
    dirName += 'isotropic/'
tic1 = clock()
# if resume:
#     mn += int(iStart * ndump / rm)

while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Define discontinuous spaces on the new mesh:
    W = VectorFunctionSpace(mesh, 'DG', 1) * FunctionSpace(mesh, 'DG', 1)
    uv_2d = Function(W.sub(0))
    elev_2d = Function(W.sub(1))

    # Enforce initial conditions on discontinuous space / load variables from disk:
    index = mn * int(rm / ndump)
    indexStr = stor.indexString(index)
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

    # Compute Hessian and metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    fAdapt = Function(W.sub(1))
    if mtype == 'f':
        fAdapt.interpolate(elev_2d)
    elif mtype == 's':
        fAdapt.interpolate(sqrt(dot(uv_2d, uv_2d)))
    if iso:
        M = adap.isotropicMetric(V, fAdapt)
    else:
        H = adap.constructHessian(mesh, V, fAdapt, method=hessMeth)
        M = adap.computeSteadyMetric(mesh, V, H, fAdapt, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)

    # Adapt mesh with respect to computed metric field and interpolate functions onto new mesh:
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)

    # Get solver parameter values and construct solver:
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.simulation_export_time = dt * ndump
    options.simulation_end_time = (mn + 1) * dt * rm
    options.timestepper_type = op.timestepper
    options.timestep = dt

    # Specify outfile directory and HDF5 checkpointing:
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
    field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
    e = exporter.ExportManager(dirName + 'hdf5',
                               ['elev_2d', 'uv_2d'],
                               field_dict,
                               field_metadata,
                               export_type='hdf5')
    solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)

    # Timestepper bookkeeping for export time step
    solver_obj.i_export = index
    solver_obj.next_export_t = solver_obj.i_export * options.simulation_export_time
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t / dt))
    solver_obj.simulation_time = solver_obj.iteration * dt

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
    nEle, nVer = adap.meshStats(mesh)
    N = [min(nEle, N[0]), max(nEle, N[1])]
    Sn += nEle
    mn += 1
    print('\n************************** Adaption step %d ****************************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (mn * rm * dt / 60., T / 60.))
    print('#Elements after adaption step %d: %d' % (mn, nEle))
    print('Min/max #Elements:', N, ' Mean #Elements: %d' % (Sn / mn))
    print('Elapsed time for this step: %1.2fs \n' % (toc2 - tic2))
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
