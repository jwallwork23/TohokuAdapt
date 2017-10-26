from firedrake import *
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


print('************** ADJOINT-BASED ADAPTIVE TSUNAMI SIMULATION **************\n')

# Define initial mesh and mesh statistics placeholders:
print('ADJOINT-GUIDED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = dom.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 4): ') or 4))
nEle, nVer = adap.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
mesh0 = mesh
W0 = VectorFunctionSpace(mesh0, 'DG', 1) * FunctionSpace(mesh0, 'DG', 1)
print('...... mesh loaded. Initial #Vertices : %d. Initial #Elements : %d.' % (nVer, nEle))

# Get default adaptivity parameter values:
op = opt.Options(vscale=0.2, mtype='f', rm=60)
numVer = op.vscale * nVer
hmin = op.hmin
hmax = op.hmax
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = op.ntype
mtype = op.mtype
beta = op.beta
iso = op.iso
if not iso:
    hessMeth = op.hessMeth

# Get physical parameters:
g = op.g

# Get solver parameters:
T = op.T
Ts = op.Ts
dt = op.dt
Dt = Constant(dt)
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print('WARNING: chosen timestep dt = %.2fs exceeds recommended value of %.2fs' % (dt, cdt))
    if bool(input('Hit anything except enter if happy to proceed.')) or False:
        exit(23)
ndump = op.ndump
rm = op.rm
stored = bool(input('Hit anything but enter if adjoint data is already stored: '))

# Establish filename:
dirName = 'plots/adjointBased/'
if iso:
    dirName += 'isotropic/'

if not stored:
    # Initalise counters:
    t = T
    mn = int(T / (rm * dt))
    dumpn = ndump
    meshn = rm
    tic1 = clock()

    # Forcing switch:
    coeff = Constant(1.)
    switch = True

    # Establish adjoint variables and apply initial conditions:
    lam_ = Function(W0)
    lu_, le_ = lam_.split()
    lu_.interpolate(Expression([0, 0]))
    le_.interpolate(Expression(0))
    lam = Function(W0)
    lam.assign(lam_)
    lu, le = lam.split()
    lu.rename('Adjoint velocity')
    le.rename('Adjoint free surface')

    # Store final time data to HDF5 and PVD:
    with DumbCheckpoint(dirName + 'hdf5/adjoint_' + stor.indexString(mn), mode=FILE_CREATE) as chk:
        chk.store(lu)
        chk.store(le)
        chk.close()
    adjointFile = File(dirName + 'adjoint.pvd')
    adjointFile.write(lu, le, time=T)

    # Establish test functions and midpoint averages:
    w, xi = TestFunctions(W0)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Establish smoothened indicator function for adjoint equations:
    f = Function(W0.sub(1), name='Forcing term')
    f.interpolate(Expression('(x[0] > 490e3) & (x[0] < 640e3) & (x[1] > 4160e3) & (x[1] < 4360e3) ? ' +
                             'exp(1. / (pow(x[0] - 565e3, 2) - pow(75e3, 2))) * ' +
                             'exp(1. / (pow(x[1] - 4260e3, 2) - pow(100e3, 2))) : 0.'))

    # Set up the variational problem:
    L = ((le - le_) * xi - Dt * g * inner(luh, grad(xi)) - coeff * f * xi
          + inner(lu - lu_, w) + Dt * (b * inner(grad(leh), w) + leh * inner(grad(b), w))) * dx
    adjointProblem = NonlinearVariationalProblem(L, lam)
    adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=op.params)

    # Split to access data:
    lu, le = lam.split()
    lu_, le_ = lam_.split()

    print('\nStarting fixed resolution adjoint run...')
    while t > 0.5 * dt:

        # Increment counters:
        t -= dt
        dumpn -= 1
        meshn -= 1

        # Modify forcing term:
        if (t < Ts + 1.5 * dt) & switch:
            coeff.assign(0.5)
        elif (t < Ts + 0.5 * dt) & switch:
            switch = False
            coeff.assign(0.)

    # Solve the problem and update:
        adjointSolver.solve()
        lam_.assign(lam)

        # Dump to vtu:
        if dumpn == 0:
            dumpn += ndump
            adjointFile.write(lu, le, time=t)

        # Dump to HDF5:
        if meshn == 0:
            meshn += rm
            mn -= 1
            # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
            if not stored:
                print('t = %1.1fs' % t)
                with DumbCheckpoint(dirName + 'hdf5/adjoint_' + stor.indexString(mn), mode=FILE_CREATE) as chk:
                    chk.store(lu)
                    chk.store(le)
                    chk.close()

    toc1 = clock()
    print('... done! Elapsed time for adjoint solver: %1.2fs' % (toc1 - tic1))
    assert(mn == 0)

# Initialise counters:
dumpn = 0   # Dump counter
mn = 0      # Mesh number
Sn = 0      # Sum over #Elements
tic1 = clock()

# Approximate isotropic metric at boundaries of initial mesh using circumradius:
h = Function(W0.sub(1))
h.interpolate(CellSize(mesh0))
M_ = adap.isotropicMetric(TensorFunctionSpace(mesh0, 'CG', 1), h, bdy=True)

print('\nStarting mesh adaptive forward run...')
while mn < np.ceil(T / (dt * rm)):
    tic2 = clock()

    # Define discontinuous spaces on the new mesh:
    mixedDG1 = VectorFunctionSpace(mesh, 'DG', 1) * FunctionSpace(mesh, 'DG', 1)
    uv_2d = Function(mixedDG1.sub(0))
    elev_2d = Function(mixedDG1.sub(1))

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

    # Create functions to hold inner product and significance data:
    ip = Function(mixedDG1.sub(1), name='Inner product')
    significance = Function(mixedDG1.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        # Read in saved data from HDF5:
        with DumbCheckpoint(dirName + 'hdf5/adjoint_' + stor.indexString(mn), mode=FILE_CREATE) as chk:
            lu = Function(W0.sub(0), name='Adjoint velocity')
            le = Function(W0.sub(1), name='Adjoint free surface')
            chk.load(lu)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if mn != 0:
            print('    #### Interpolating adjoint data')
            lu, le = interp(mesh, lu, le)

        # Multiply fields together:
        ip.dat.data[:] = lu.dat.data[:, 0] * uv_2d.dat.data[:, 0] + lu.dat.data[:, 1] * uv_2d.dat.data[:, 1]
        ip.dat.data[:] += le.dat.data * elev_2d.dat.data

        # Extract (pointwise) maximal values:
        if j == 0:
            significance.dat.data[:] = ip.dat.data[:]
        else:
            for k in range(len(ip.dat.data)):
                if np.abs(ip.dat.data[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = ip.dat.data[k]

    # Interpolate initial mesh size onto new mesh and build associated boundary metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    M_ = adap.isotropicMetric(V, inte.interp(mesh, h)[0], bdy=True)

    # Generate metric associated with significant data:
    if iso:
        M = adap.isotropicMetric(V, significance)
    else:
        H = Function(V)
        if mtype == 's':
            spd = Function(W.sub(1))
            spd.interpolate(sqrt(dot(u, u)))
            H = adap.constructHessian(mesh, V, spd, method=hessMeth)
        elif mtype == 'f':
            H = adap.constructHessian(mesh, V, eta, method=hessMeth)
        else:
            raise NotImplementedError('Cannot currently perform adjoint-based adaption with respect to two fields.')
        for k in range(mesh.topology.num_vertices()):
            H.dat.data[k] *= significance.dat.data[k]
        M = adap.computeSteadyMetric(mesh, V, H, eta, h_min=hmin, h_max=hmax, normalise=ntype, num=numVer)

    # Gradate metric, adapt mesh and interpolate variables:
    M = adap.metricIntersection(mesh, V, M, M_, bdy=True)
    adap.metricGradation(mesh, M, beta, isotropic=iso)
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
    significanceFile.write(significance, time=solver_obj.simulation_time)

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
    print('\n************ Adaption step %d **************' % mn)
    print('Time = %1.2f mins / %1.1f mins' % (mn * rm * dt / 60., T / 60.))
    print('Number of vertices after adaption step %d: ' % mn, nEle)
    print('#Elements after adaption step %d: %d' % (mn, nEle))
    print('Min/max #Elements:', N, ' Mean #Elements: %d' % (Sn / mn))
    print('Elapsed time for this step: %1.2fs \n' % (toc2 - tic2))
toc1 = clock()
print('Elapsed time for forward solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
