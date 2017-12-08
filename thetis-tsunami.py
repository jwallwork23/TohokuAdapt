from thetis import *
from thetis.field_defs import field_metadata
from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.forms as form
import utils.interpolation as inte
import utils.mesh as msh
import utils.options as opt


# Define initial mesh and mesh statistics placeholders
print('*********************** TOHOKU TSUNAMI SIMULATION *********************\n')
approach = input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': ") or 'simpleAdapt'
op = opt.Options(vscale=0.4 if approach == 'goalBased' else 0.85,
                 rm=60 if approach == 'goalBased' else 30,
                 gradate=False,
                 advect=False,
                 coarseness=4)
mesh, eta0, b = msh.TohokuDomain(op.coarseness)
dirName = 'plots/' + approach + '/' + msh.MeshSetup(op.coarseness).meshName + '/'
msh.saveMesh(mesh, dirName + 'hdf5/mesh')

# Get solver parameters
T = op.Tend
dt = op.dt
Dt = Constant(dt)
op.checkCFL(b)
ndump = op.ndump

# Get adaptivity parameters
iso = op.iso
if iso:
    dirName += 'isotropic/'
hmin = op.hmin
hmax = op.hmax
rm = op.rm
nEle, nVer = msh.meshStats(mesh)
N = [nEle, nEle]            # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices

# Initialise counters
cnt = 0

if approach in ('fixedMesh', 'goalBased'):

    if approach == 'goalBased':
        # Create FunctionSpace upon which to define forms and residuals
        V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
        P0 = FunctionSpace(mesh, "DG", 0)
        v = TestFunction(P0)

        # Define Functions to hold residual and adjoint solution data
        rho = Function(V)
        rho_u, rho_e = rho.split()
        rho_u.rename("Velocity residual")
        rho_e.rename("Elevation residual")
        dual = Function(V)
        dual_u, dual_e = dual.split()
        dual_u.rename("Adjoint velocity")
        dual_e.rename("Adjoint elevation")

    # Get solver parameter values and construct solver, with default dg1-dg1 space
    tic = clock()
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = False
    options.use_grad_depth_viscosity_term = False
    options.simulation_export_time = op.dt * op.ndump
    options.simulation_end_time = op.T
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

    # Apply ICs and time integrate
    solver_obj.assign_initial_conditions(elev=eta0)
    solver_obj.iterate()
    fixedMeshTime = clock() - tic
    print('Time elapsed for fixed mesh solver: %.1fs (%.2fmins)' % (fixedMeshTime, fixedMeshTime / 60))

    # TODO: somehow integrate this at EACH TIMESTEP:

    # if approach == 'goalBased':
    #     # Tell dolfin about timesteps, so it can compute functionals including measures of time other than dt[FINISH_TIME]
    #     if t >= T - dt:
    #         finished = True
    #     if t == 0.:
    #         adj_start_timestep()
    #     else:
    #         adj_inc_timestep(time=t, finished=finished)
    #
    #     if not cnt % rm:
    #         # Approximate residual of forward equation and save to HDF5
    #         Au, Ae = form.strongResidualSW(q, q_, b, Dt)
    #         rho_u.interpolate(Au)
    #         rho_e.interpolate(Ae)
    #         with DumbCheckpoint(dirName + 'hdf5/residual_' + op.indexString(cnt), mode=FILE_CREATE) as chk:
    #             chk.store(rho_u)
    #             chk.store(rho_e)
    #             chk.close()
    #
    #         # Print to screen, save data and increment counters
    #         residualFile.write(rho_u, rho_e, time=t)
    #
    #     cnt += 1


# TODO: adjoint run, using included timesteps.
# TODO: load residual data and calculate error estimators. Save these data.

if approach in ('simpleAdapt', 'goalBased'):
    mn = 0
    tic1 = clock()
    while mn < np.ceil(T / (dt * rm)):
        tic2 = clock()
        index = mn * int(rm / ndump)
        W = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
        elev_2d, uv_2d = op.loadFromDisk(W, index, dirName, elev0=eta0)  # Enforce ICs / load variables from disk

        # Compute Hessian and metric, adapt mesh and interpolate variables
        V = TensorFunctionSpace(mesh, 'CG', 1)
        if op.mtype != 's':
            if iso:
                M = adap.isotropicMetric(V, elev_2d, op=op)
            else:
                H = adap.constructHessian(mesh, V, elev_2d, op=op)
                M = adap.computeSteadyMetric(mesh, V, H, elev_2d, nVerT=nVerT, op=op)

        # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
        if approach == 'goalBased':
            with DumbCheckpoint(dirName + 'hdf5/error_' + op.indexString(cnt), mode=FILE_READ) as loadError:
                loadError.load(epsilon)  # P0 field on the initial mesh
                loadError.close()
            errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
            for k in range(mesh.topology.num_vertices()):
                H.dat.data[k] *= errEst.dat.data[k]  # Scale by error estimate

        if op.mtype != 'f':
            spd = Function(W.sub(1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
            if iso:
                M2 = adap.isotropicMetric(V, spd, op=op)
            else:
                H = adap.constructHessian(mesh, V, spd, op=op)
                M2 = adap.computeSteadyMetric(mesh, V, H, spd, nVerT=nVerT, op=op)
            M = adap.metricIntersection(mesh, V, M, M2) if op.mtype == 'b' else M2
        if op.gradate:
            adap.metricGradation(mesh, M)
        if op.advect & mn != 0:
            adap.advectMetric(M, uv_2d, dt, rm)  # Advect metric ahead in direction of velocity
        mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
        elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)
        msh.saveMesh(mesh, dirName + 'hdf5/mesh_' + op.indexString(index))  # Save mesh to disk

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
        e = exporter.ExportManager(dirName + 'hdf5', ['elev_2d', 'uv_2d'], field_dict, field_metadata,
                                   export_type='hdf5')
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

# TODO: test ``simpleAdapt`` script
