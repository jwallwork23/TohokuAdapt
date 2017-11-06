from thetis import *
from thetis.field_defs import field_metadata

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.interp as inte

print('\n******************************** SHALLOW WATER TEST PROBLEM ********************************\n')
print('Mesh adaptive solver initially defined on a square mesh')

# Set parameter values
depth = 0.1
T = 2.
g = 9.81
dt = 0.05
Dt = Constant(dt)
ndump = 1       # Timesteps per data dump
rm = 5          # Timesteps per simpleAdapt remeshes
# TODO: check CFL criterion
dirName = 'plots/tests/simpleAdapt'

# Define inital mesh and function space
n = 100
lx = 2 * np.pi
mesh = SquareMesh(n, n, lx, lx)
x, y = SpatialCoordinate(mesh)
W = VectorFunctionSpace(mesh, "DG", 1) * FunctionSpace(mesh, "CG", 2)
b = Function(W.sub(1), name="Bathymetry").assign(depth)
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('Initial #Elements : %d. Initial #Vertices : %d. \n' % (nEle, nVer))

# Initialise counters
t = 0.
dumpn = 0
mn = 0
Sn = 0
tic1 = clock()

# TODO: run fixedMesh solver forward in time to t = T, storing to hdf5
# TODO: run fixed mesh adjoint solver backward in time to t = 0, again storing to hdf5

while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Enforce initial conditions on discontinuous space / load variables from disk
    W = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    uv_2d = Function(W.sub(0))
    elev_2d = Function(W.sub(1))
    index = mn * int(rm / ndump)
    indexStr = stor.indexString(index)
    if mn == 0:
        elev_2d.interpolate(1e-3 * exp(- (pow(x - 2., 2) + pow(y - 2., 2))))
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
        if op.mtype != 's':
            M = adap.isotropicMetric(V, elev_2d, op=op) if iso else adap.computeSteadyMetric(
                mesh, V, adap.constructHessian(mesh, V, elev_2d, op=op), elev_2d, nVerT=nVerT, op=op)
        if op.mtype != 'f':
            spd = Function(W.sub(1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
            M2 = adap.isotropicMetric(V, spd) if iso else adap.computeSteadyMetric(
                mesh, V, adap.constructHessian(mesh, V, spd, op=op), spd, nVerT=nVerT, op=op)
            M = adap.metricIntersection(mesh, V, M, M2) if op.mtype == 'b' else M2
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

# TODO: run adjointBased solver