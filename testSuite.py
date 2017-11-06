from thetis import *
from thetis.field_defs import field_metadata

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.interpolation as inte
import utils.options as opt
import utils.storage as stor

print('\n******************************** SHALLOW WATER TEST PROBLEM ********************************\n')
print('Mesh adaptive solver initially defined on a square mesh')

# Define inital mesh and function space
n = 64
lx = 2 * np.pi
mesh = SquareMesh(n, n, lx, lx)
x, y = SpatialCoordinate(mesh)
W = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "CG", 2)
b = Function(W.sub(1), name="Bathymetry").assign(0.1)
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('... with #Elements : %d. Initial #Vertices : %d. \n' % (nEle, nVer))

# Set parameter values
op = opt.Options(dt=0.05, hmin=5e-2, hmax=1., T=2., ndump=1)
nVerT = op.vscale * nVer    # Target #Vertices
T = op.T
dt = op.dt
op.checkCFL(b)
ndump = op.ndump
rm = 5      # Timesteps per simpleAdapt remeshes
dirName = 'plots/tests/simpleAdapt/'

# Initialise counters
t = 0.
dumpn = 0
mn = 0
Sn = 0
tic1 = clock()

# TODO: run fixedMesh solver forward in time to t = T, storing to hdf5
# TODO: run fixed mesh adjoint solver backward in time to t = 0, again storing to hdf5
# TODO: run adjointBased solver

while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Enforce initial conditions on discontinuous space / load variables from disk
    W = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    uv_2d = Function(W.sub(0))
    elev_2d = Function(W.sub(1))
    index = mn * int(rm / ndump)
    indexStr = stor.indexString(index)
    if mn == 0:
        elev_2d.interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
        uv_2d.interpolate(Expression((0, 0)))
    else:
        with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as el:
            el.load(elev_2d, name='elev_2d')
            el.close()
        with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as ve:
            ve.load(uv_2d, name='uv_2d')
            ve.close()

    # Compute Hessian and metric, adapt mesh and interpolate variables
    if mn != 0:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        M = adap.computeSteadyMetric(mesh, V, adap.constructHessian(mesh, V, elev_2d, op=op),
                                     elev_2d, nVerT=nVerT, op=op)
        mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
        elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)

    # Get solver parameter values and construct solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = False
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

    # Timestepper bookkeeping for export time step and next export
    solver_obj.i_export = index
    solver_obj.next_export_t = solver_obj.i_export * options.simulation_export_time
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t / dt))
    solver_obj.simulation_time = solver_obj.iteration * dt
    solver_obj.next_export_t += options.simulation_export_time
    for e in solver_obj.exporters.values():
        e.set_next_export_ix(solver_obj.i_export)

    # Time integrate and print
    solver_obj.iterate()
    nEle, nVer = adap.meshStats(mesh)
    N = [min(nEle, N[0]), max(nEle, N[1])]
    Sn += nEle
    mn += 1
    print("""\n************************** Adaption step %d ****************************
Percent complete  : %4.1f%%    Elapsed time : %4.2fs (This step : %4.2fs)     
#Elements... Current : %d  Mean : %d  Minimum : %s  Maximum : %s\n""" %
          (mn, (100 * mn * rm * dt) / T, clock() - tic1, clock() - tic2, nEle, Sn / mn, N[0], N[1]))
toc1 = clock()
print('Elapsed time for simple adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
