from firedrake import *
from thetis import *
from thetis.field_defs import field_metadata

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.error as err
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
eta0 = Function(W.sub(1), name="Initial surface").interpolate(1e-3 * exp(-(pow(x - np.pi, 2) + pow(y - np.pi, 2))))
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('... with #Elements : %d. Initial #Vertices : %d. \n' % (nEle, nVer))

# Set parameter values
op = opt.Options(dt=0.05, hmin=5e-2, hmax=1., T=2., ndump=1)
nVerT = op.vscale * nVer    # Target #Vertices
T = op.T
g = op.g
dt = op.dt
Dt = Constant(dt)
op.checkCFL(b)
ndump = op.ndump

# Save initial function space
mesh0 = mesh
W0 = VectorFunctionSpace(mesh0, op.space1, op.degree1) * FunctionSpace(mesh0, op.space2, op.degree2)

# Run fixedMesh forward solver
print('******************** FIXED MESH SHALLOW WATER TEST ********************\n')
tic1 = clock()
dirName = "plots/tests/fixedMesh"
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.element_family = op.family
options.use_nonlinear_equations = False
options.simulation_export_time = dt * ndump
options.simulation_end_time = T
options.timestepper_type = op.timestepper
options.timestep = dt
options.output_directory = dirName
options.export_diagnostics = True
options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
solver_obj.assign_initial_conditions(elev=eta0)
solver_obj.iterate()
fixedMeshTime = clock() - tic1
print('Elapsed time for fixed mesh solver: %1.1fs (%1.2f mins) \n' % (fixedMeshTime, fixedMeshTime / 60))

print('************ "SIMPLE" MESH ADAPTIVE SHALLOW WATER TEST ****************\n')
rm = 5
dirName = 'plots/tests/simpleAdapt/'
t = 0.
mn = 0
Sn = 0
tic1 = clock()

while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Enforce initial conditions on discontinuous space / load variables from disk
    W = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    uv_2d = Function(W.sub(0))
    elev_2d = Function(W.sub(1))
    index = mn * int(rm / ndump)
    indexStr = stor.indexString(index)
    if mn == 0:
        elev_2d.assign(eta0)
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
simpleAdaptTime = clock() - tic1
print('Elapsed time for simple adaptive solver: %1.1fs (%1.2f mins) \n' % (simpleAdaptTime, simpleAdaptTime / 60))

print('********** ADJOINT BASED MESH ADAPTIVE SHALLOW WATER TEST *************\n')
rm = 10
t = T
mn = int(T / dt)
tic1 = clock()

# Solver parameters
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Establish adjoint variables and apply initial conditions
lam_ = Function(W0)
lu_, le_ = lam_.split()
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression(0))
lam = Function(W0).assign(lam_)
lu, le = lam.split()
lu.rename("Adjoint velocity")
le.rename("Adjoint free surface")
adjointFile = File("plots/tests/adjointBased/adjoint.pvd")
b = Function(W0.sub(1), name="Bathymetry").assign(0.1)

# Establish (smoothened) indicator function for adjoint equations
x1 = 0.
x2 = 0.4
y1 = np.pi - 0.4
y2 = np.pi + 0.4
fexpr = "(x[0] >= %.2f) & (x[0] < %.2f) & (x[1] > %.2f) & (x[1] < %.2f) ? 1e-3 : 0." % (x1, x2, y1, y2)
f = Function(W0.sub(1), name="Forcing term").interpolate(Expression(fexpr))

# Set up the variational problem, using Crank Nicolson timestepping
w, xi = TestFunctions(W0)
lu, le = split(lam)
lu_, le_ = split(lam_)
L = ((le - le_) * xi + inner(lu - lu_, w)
     - Dt * g * inner(0.5 * (lu + lu_), grad(xi)) - f * xi
     + Dt * (b * inner(grad(0.5 * (le + le_)), w) + 0.5 * (le + le_) * inner(grad(b), w))) * dx
adjointProblem = NonlinearVariationalProblem(L, lam)
adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=params)
lu, le = lam.split()
lu_, le_ = lam_.split()

print("Starting adjoint run...")
while mn > 0:
    print("t = %5.2fs" % t)

    # Solve the problem, update variables and dump to vtu and HDF5
    if mn != int(T / dt):
        adjointSolver.solve()
        lam_.assign(lam)
    adjointFile.write(lu, le, time=t)
    with DumbCheckpoint("plots/tests/adjointBased/hdf5/adjoint_" + stor.indexString(mn), mode=FILE_CREATE) as chk:
        chk.store(lu)
        chk.store(le)
        chk.close()
    t -= dt
    mn -= 1
assert(mn == 0)
adjointRunTime = clock() - tic1
print('Elapsed time for fixed mesh adjoint solver: %1.1fs (%1.2f mins) \n' % (adjointRunTime, adjointRunTime / 60))

# Set up forward problem
iStart = int(op.Ts / (dt * rm))         # Index corresponding to tStart
iEnd = int(np.ceil(T / (dt * rm)))      # Index corresponding to tEnd
dirName = 'plots/tests/adjointBased'

print("Starting forward run...")
while mn < iEnd:
    tic2 = clock()

    # Enforce initial conditions on discontinuous space / load variables from disk
    mixedSpace = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    uv_2d = Function(mixedSpace.sub(0))
    elev_2d = Function(mixedSpace.sub(1))
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

    # Create functions to hold inner product and significance data
    spaceMatch = (op.space1 == op.space2) & (op.degree1 == op.degree2)
    W = mixedSpace.sub(1) if spaceMatch else FunctionSpace(mesh, op.space1, op.degree1)
    significance = Function(W, name='Significant regions')

    for j in range(max(mn, iStart), iEnd):
        # Load adjoint data and interpolate onto current mesh
        with DumbCheckpoint(dirName + 'hdf5/adjoint_' + stor.indexString(mn), mode=FILE_READ) as chk:
            lu = Function(W0.sub(0), name='Adjoint velocity')
            le = Function(W0.sub(1), name='Adjoint free surface')
            chk.load(lu)
            chk.load(le)
            chk.close()
        if mn != 0:
            print('#### Interpolating adjoint data...')
            print('    #### Step %d / %d' % (j, iEnd - max(mn, iStart)))
            lu, le = inte.interp(mesh, lu, le)

        # Estimate error and extract (pointwise) maximal values
        rho = err.basicErrorEstimator(uv_2d, lu, elev_2d if spaceMatch else Function(W).interpolate(elev_2d),
                                      le if spaceMatch else Function(W).interpolate(le)).dat.data
        if j == 0:
            significance.dat.data[:] = rho
        else:
            for k in range(len(rho)):
                if np.abs(rho[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = rho[k]

    # Interpolate initial mesh size onto new mesh and build associated boundary metric
    V = TensorFunctionSpace(mesh, 'CG', 1)
    M_ = adap.isotropicMetric(V, inte.interp(mesh, h)[0], bdy=True, op=op)

    # Generate metric associated with significant data, gradate it, adapt mesh and interpolate variables
    H = Function(V)
    H = adap.constructHessian(mesh, V, elev_2d, op=op)
    for k in range(mesh.topology.num_vertices()):
        H.dat.data[k] *= significance.dat.data[k]
    M = adap.computeSteadyMetric(mesh, V, H, elev_2d, nVerT=nVerT, op=op)
    M = adap.metricIntersection(mesh, V, M, M_, bdy=True)
    adap.metricGradation(mesh, M, op.beta)
    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
    elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)

    # Get solver parameter values and construct solver, using a P1DG-P2 mixed function space
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

    # Timestepper bookkeeping for export time step
    solver_obj.i_export = index
    solver_obj.next_export_t = solver_obj.i_export * options.simulation_export_time
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t / dt))
    solver_obj.simulation_time = solver_obj.iteration * dt

    # For next export
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
adjointBasedRunTime = clock() - tic1
print('Elapsed time for adjoint based solver: %1.1fs (%1.2f mins)' % (adjointBasedRunTime, adjointBasedRunTime / 60))

# TODO: use firedrake-adjoint or create a custom modified solver_obj for adjoint LSWEs in Thetis
# TODO: use explicit error estimators

print("""\n************************* Times to solution ****************************
Fixed mesh solver           %5.2fs  Simple adaptive solver      %5.2fs
Fixed mesh adjoint solver   %5.2fs  Adjoint based solver        %5.2fs"""
      % (fixedMeshTime, simpleAdaptTime, adjointRunTime, adjointBasedRunTime))
