from thetis import *
from thetis.field_defs import field_metadata

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt


# Define initial mesh and mesh statistics placeholders
print('*************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ***************\n')
print('MESH ADAPTIVE solver initially defined on a mesh of')
mesh, eta0, b = msh.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 5): ') or 5))
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]            # Min/max #Elements
print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d. \n' % (nEle, nVer))

# Get default parameter values and check CFL criterion
op = opt.Options(outputHessian=True)
nVerT = op.vscale * nVer    # Target #Vertices
dirName = 'plots/simpleAdapt/'
T = op.T
dt = op.dt
op.checkCFL(b)
ndump = op.ndump
rm = op.rm

# Initialise counters:
dumpn = mn = Sn = 0   # Dump counter, mesh number and sum over #Elements
tic1 = clock()
hfile = File(dirName + "hessian.pvd")

while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()
    index = mn * int(rm / ndump)
    W = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    elev_2d, uv_2d = op.loadFromDisk(W, index, dirName, elev0=eta0)     # Enforce ICs / load variables from disk

    # Compute Hessian and metric, adapt mesh and interpolate variables
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if op.mtype != 's':
        H = adap.constructHessian(mesh, V, elev_2d, op=op)
        M = adap.computeSteadyMetric(mesh, V, H, elev_2d, nVerT=nVerT, op=op)
    if op.mtype != 'f':
        spd = Function(W.sub(1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
        H = adap.constructHessian(mesh, V, spd, op=op)
        M2 = adap.computeSteadyMetric(mesh, V, H, spd, nVerT=nVerT, op=op)
        M = adap.metricIntersection(mesh, V, M, M2) if op.mtype == 'b' else M2
    adap.advectMetric(mesh, M, uv_2d, 5e4, pm=-1)
    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
    elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)
    if op.outputHessian:
        H.rename("Hessian")
        hfile.write(H, time=float(mn))

    # Establish Thetis flow solver object
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = False
    options.use_grad_depth_viscosity_term = False
    options.simulation_export_time = dt * ndump
    options.simulation_end_time = (mn + 1) * dt * rm
    options.timestepper_type = op.timestepper
    options.timestep = dt
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
    solver_obj.next_export_t += options.simulation_export_time  # For next export
    for e in solver_obj.exporters.values():
        e.set_next_export_ix(solver_obj.i_export)

    # Time integrate and print
    solver_obj.iterate()
    nEle = msh.meshStats(mesh)[0]
    N = [min(nEle, N[0]), max(nEle, N[1])]
    Sn += nEle
    mn += 1
    op.printToScreen(mn, clock() - tic1, clock() - tic2, nEle, Sn, N)
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
