from firedrake import *
from thetis import *
from thetis.field_defs import field_metadata

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.error as err
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt
import utils.storage as stor


# Define initial mesh and mesh statistics placeholders
print('************** ADJOINT-BASED ADAPTIVE TSUNAMI SIMULATION **************\n')
print('ADJOINT-GUIDED mesh adaptive solver initially defined on a mesh of')
mesh, eta0, b = msh.TohokuDomain(int(input('coarseness (Integer in range 1-5, default 4): ') or 4))
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]    # Min/max #Elements
print('...... mesh loaded. Initial #Elements : %d. Initial #Vertices : %d.' % (nEle, nVer))

# Get default parameter values and check CFL criterion
op = opt.Options(vscale=0.2, rm=60, ndump=1, outputHessian=True, mtype='b') # ndump=1 needed for explicit error estn
nVerT = op.vscale * nVer                                                    # Target #Vertices
dirName = 'plots/adjointBased/'
basic = bool(input('Hit anything but enter to use basic error estimators, as in DL16: '))
if not basic:
    dirName += 'explicit/'
iso = op.iso
if iso:
    dirName += 'isotropic/'
T = op.Tend
dt = op.dt
Dt = Constant(dt)
op.checkCFL(b)
ndump = op.ndump
rm = op.rm

# Initialise counters, constants and function space
Sn = meshn = 0                          # Sum over #Elements and mesh counter
iStart = int(op.Tstart / (dt * rm))     # Index corresponding to start time
iEnd = int(np.ceil(T / (dt * rm)))      # Index corresponding to end time
mesh0 = mesh
W0 = VectorFunctionSpace(mesh0, op.space1, op.degree1) * FunctionSpace(mesh0, op.space2, op.degree2)

if bool(input('Hit anything but enter if adjoint data is already stored: ')):
    mn = 0
else:
    # Initalise counters and forcing switch
    t = T
    mn = iEnd
    tic1 = clock()
    coeff = Constant(1.)
    switch = True

    # Establish adjoint variables and apply initial conditions
    lam_ = Function(W0)
    lu_, le_ = lam_.split()
    lu_.interpolate(Expression([0, 0]))
    le_.interpolate(Expression(0))
    lam = Function(W0).assign(lam_)
    lu, le = lam.split()            # Velocity and free surface components of adjoint variable, respectively
    lu.rename('Adjoint velocity')
    le.rename('Adjoint free surface')

    # Establish (smoothened) indicator function for adjoint equations
    fexpr = '(x[0] > 490e3) & (x[0] < 640e3) & (x[1] > 4160e3) & (x[1] < 4360e3) ? ' \
            'exp(1. / (pow(x[0] - 565e3, 2) - pow(75e3, 2))) * exp(1. / (pow(x[1] - 4260e3, 2) - pow(100e3, 2))) : 0.'
    f = Function(W0.sub(1)).interpolate(Expression(fexpr))

    # Set up the variational problem, using Crank Nicolson timestepping
    w, xi = TestFunctions(W0)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    L = ((le - le_) * xi + inner(lu - lu_, w)
         - Dt * op.g * inner(0.5 * (lu + lu_), grad(xi)) - coeff * f * xi
         + Dt * (b * inner(grad(0.5 * (le + le_)), w) + 0.5 * (le + le_) * inner(grad(b), w))) * dx
    adjointProblem = NonlinearVariationalProblem(L, lam)
    adjointSolver = NonlinearVariationalSolver(adjointProblem, solver_parameters=op.params)
    lu, le = lam.split()
    lu_, le_ = lam_.split()

    print('\nStarting fixed resolution adjoint run...')
    while t > 0.5 * dt:

        # Modify forcing term
        if (t < op.Tstart + 1.5 * dt) & switch:
            coeff.assign(0.5)
        elif (t < op.Tstart + 0.5 * dt) & switch:
            switch = False
            coeff.assign(0.)

        # Solve the problem, update variables and dump to HDF5
        if t != T:
            adjointSolver.solve()
            lam_.assign(lam)
        if meshn == 0:
            print('t = %1.1fs' % t)
            with DumbCheckpoint(dirName + 'hdf5/adjoint_' + op.indexString(mn), mode=FILE_CREATE) as chk:
                chk.store(lu)
                chk.store(le)
                chk.close()
            meshn += rm
            mn -= 1
        t -= dt
        meshn -= 1
    print('... done! Elapsed time for adjoint solver: %1.2fs' % (clock() - tic1))
    assert(mn == 0)

# Approximate isotropic metric at boundaries of initial mesh using circumradius
h = Function(W0.sub(1)).interpolate(CellSize(mesh0))
hfile = File(dirName + "hessian.pvd")
sfile = File(dirName + "significance.pvd")

print('\nStarting mesh adaptive forward run...')
tic1 = clock()
while mn < iEnd:
    tic2 = clock()
    index = mn * int(rm / ndump)
    i0 = max(mn, iStart)
    Wcomp = FunctionSpace(mesh, "CG", 1)                                # Computational space to match metric P1 space
    W = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    elev_2d, uv_2d = op.loadFromDisk(W, index, dirName, elev0=eta0)     # Enforce ICs / load variables from disk
    if not basic:
        hk = Function(Wcomp).interpolate(CellSize(mesh))                # Current sizes of mesh elements
    if (not basic) and (mn != 0):
        print('#### Interpolating adjoint data...')
        elev_2d_, uv_2d_ = op.loadFromDisk(W, index - 1, dirName, elev0=eta0)    # Load saved forward data
    for j in range(i0, iEnd):
        le, lu = op.loadFromDisk(W0, j, dirName, adjoint=True)          # Load saved adjoint data
        if mn != 0:
            print('    #### Step %d / %d' % (j + 1 - i0, iEnd - i0))
            lu, le = inte.interp(mesh, lu, le)                                      # Interpolate onto current mesh
        rho = err.basicErrorEstimator(uv_2d, lu, elev_2d, le) if (basic or mn == 0) else \
            err.explicitErrorEstimator(uv_2d_, uv_2d, elev_2d_, elev_2d, lu, le, b, dt, hk)     # Estimate error
        if j == i0:
            errEst = Function(Wcomp).assign(rho)
        else:
            errEst = adap.pointwiseMax(errEst, rho)     # Extract (pointwise) maximal values
    errEst = assemble(sqrt(errEst * errEst))            # Take modulus (error estimator must be strictly positive)
    errEst.rename("Local error indicators")             # TODO: how to use the ufl function `abs`?
    sfile.write(errEst)

    V = TensorFunctionSpace(mesh, "CG", 1)
    M_ = adap.isotropicMetric(V, inte.interp(mesh, h)[0], bdy=True, op=op)      # Initial boundary metric
    if op.mtype != 's':
        if iso:
            M = adap.isotropicMetric(V, elev_2d, op=op)
            for k in range(mesh.topology.num_vertices()):
                M.dat.data[k] *= errEst.dat.data[k]
        else:
            H = adap.constructHessian(mesh, V, elev_2d, op=op)
            for k in range(mesh.topology.num_vertices()):
                H.dat.data[k] *= errEst.dat.data[k]                             # Scale by error estimate
            M = adap.computeSteadyMetric(mesh, V, H, elev_2d, nVerT=nVerT, op=op)
    if op.mtype != 'f':
        spd = Function(W.sub(1)).interpolate(sqrt(dot(uv_2d, uv_2d)))           # Interpolate fluid speed
        if iso:
            M2 = adap.isotropicMetric(V, spd, op=op)
            for k in range(mesh.topology.num_vertices()):
                M2.dat.data[k] *= errEst.dat.data[k]
        else:
            H = adap.constructHessian(mesh, V, spd, op=op)
            for k in range(mesh.topology.num_vertices()):
                H.dat.data[k] *= errEst.dat.data[k]
            M2 = adap.computeSteadyMetric(mesh, V, H, spd, nVerT=nVerT, op=op)
        M = adap.metricIntersection(mesh, V, M, M2) if op.mtype == 'b' else M2
    M = adap.metricIntersection(mesh, V, M, M_, bdy=True)                       # Intersect with initial bdy metric
    adap.metricGradation(mesh, M, op.beta, iso=op.iso)                          # Gradate to 'smoothen' metric
    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh                          # Adapt mesh
    elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)
    if (not iso and op.outputHessian):
        H.rename("Hessian")
        hfile.write(H, time=float(mn))
    msh.saveMesh(mesh, dirName + 'hdf5/mesh_' + stor.indexString(index))  # Save mesh to disk for timeseries analysis

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

    # Timestepper bookkeeping for export time step
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
print('Elapsed time for forward solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
