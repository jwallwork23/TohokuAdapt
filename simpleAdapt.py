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


# Define initial mesh and mesh statistics placeholders
print('*************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ***************\n')
print('MESH ADAPTIVE solver initially defined on a mesh of')
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 5): ') or 5))
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d. \n' % (nEle, nVer))
# TODO: build in functionality for loading a previous state. NOTE this may involve saving the mesh to disk...
# resume = input('Hit anything except enter to resume a previous simulation.')
# if resume:
#     iStart = int(input('Simulation starting index (default i = 0)?: ')) or 0

# Get default parameter values and check CFL criterion
op = opt.Options()
nVerT = op.vscale * nVer    # Target #Vertices
iso = op.iso
dirName = 'plots/simpleAdapt/'
if iso:
    dirName += 'isotropic/'
T = op.T
dt = op.dt
cdt = op.hmin / np.sqrt(op.g * max(b.dat.data))
if dt > cdt:
    print('WARNING: chosen timestep dt = %.2fs exceeds recommended value of %.2fs' % (dt, cdt))
    if input('Hit anything except enter if happy to proceed.'):
        exit(23)
ndump = op.ndump
rm = op.rm

# Initialise counters:
dumpn = 0   # Dump counter
mn = 0      # Mesh number
Sn = 0      # Sum over #Elements
tic1 = clock()
# if resume:
#     mn += int(iStart * ndump / rm)

while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Enforce initial conditions on discontinuous space / load variables from disk
    W = VectorFunctionSpace(mesh, 'DG', 1) * FunctionSpace(mesh, 'DG', 1)
    uv_2d = Function(W.sub(0))
    elev_2d = Function(W.sub(1))
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

    # Compute Hessian and metric, adapt mesh and interpolate variables
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if op.mtype != 's':
        if iso:
            M = adap.isotropicMetric(V, elev_2d, op=op)
        else:
            H = adap.constructHessian(mesh, V, elev_2d, op=op)
            M = adap.computeSteadyMetric(mesh, V, H, elev_2d, nVerT=nVerT, op=op)
    if op.mtype != 'f':
        spd = Function(W.sub(1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
        if iso:
            M2 = adap.isotropicMetric(V, spd)
        else:
            H = adap.constructHessian(mesh, V, elev_2d, op=op)
            M2 = adap.computeSteadyMetric(mesh, V, H, spd, nVerT=nVerT, op=op)
        if op.mtype == 'b':
            M = adap.metricIntersection(mesh, V, M, M2)
        else:
            M = M2
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)

    # Get solver parameter values and construct solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.simulation_export_time = dt * ndump
    options.simulation_end_time = (mn + 1) * dt * rm
    options.timestepper_type = op.timestepper
    options.timestep = dt

    # Specify outfile directory and HDF5 checkpointing
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
    field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
    e = exporter.ExportManager(dirName + 'hdf5', ['elev_2d', 'uv_2d'], field_dict, field_metadata, export_type='hdf5')
    solver_obj.assign_initial_conditions(elev=elev_2d, uv=uv_2d)

    # Timestepper bookkeeping for export time step
    solver_obj.i_export = index
    solver_obj.next_export_t = solver_obj.i_export * options.simulation_export_time
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t / dt))
    solver_obj.simulation_time = solver_obj.iteration * dt

    # For next export
    solver_obj.export_initial_state = (dirName != options.output_directory)
    offset = 0 if solver_obj.export_initial_state else 1
    solver_obj.next_export_t += options.simulation_export_time
    for e in solver_obj.exporters.values():
        e.set_next_export_ix(solver_obj.i_export + offset)

    # Time integrate and print
    solver_obj.iterate()
    nEle, nVer = adap.meshStats(mesh)
    N = [min(nEle, N[0]), max(nEle, N[1])]
    Sn += nEle
    mn += 1
    print('\n************************** Adaption step %d ****************************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (mn * rm * dt / 60., T / 60.))
    print('#Elements after adaption step %d: %d' % (mn, nEle))
    print('Min/max #Elements:', N, ' Mean #Elements: %d' % (Sn / mn))
    print('Elapsed time for this step: %1.2fs \n' % (clock() - tic2))
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
