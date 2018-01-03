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
import utils.misc as msc
import utils.options as opt
import utils.timeseries as tim


print('*********************** TOHOKU TSUNAMI SIMULATION *********************\n')
approach, getData, getError = msc.cheatCodes(input("Choose approach: 'fixedMesh', 'simpleAdapt' or 'goalBased': "))
useAdjoint = approach == 'goalBased'
tAdapt = True

# Define initial mesh and mesh statistics placeholders
op = opt.Options(vscale=0.4 if useAdjoint else 0.85,
                 rm=60 if useAdjoint else 30,
                 gradate=True if useAdjoint else False,
                 advect=False,
                 outputHessian=False,
                 plotpvd=False,
                 coarseness=8,
                 iso=False,
                 gauges=True)
nEle = (691750, 450386, 196560, 33784, 20724, 14228, 11020, 8782, 6176)[op.coarseness]
# TODO: bootstrap to establish initial mesh resolution

# Establish filenames
dirName = 'plots/thetis-tsunami/'
if approach in ('simpleAdapt', 'goalBased'):
    dirName += approach + '/'
if op.plotpvd:
    residualFile = File(dirName + "residual.pvd")
    errorFile = File(dirName + "errorIndicator.pvd")
    adaptiveFile = File(dirName + "goalBased.pvd") if useAdjoint else File(dirName + "simpleAdapt.pvd")
if op.outputHessian:
    hessianFile = File(dirName + "hessian.pvd")

# Generate Mesh(es)
mesh, eta0, b = msh.TohokuDomain(nEle)
if useAdjoint:
    try:
        assert op.coarseness != 1
    except:
        raise NotImplementedError("Requested mesh resolution not yet available.")
    mesh_N, b_N = msh.TohokuDomain(op.coarseness-1)[0::2]   # Get finer mesh and associated bathymetry
    V_N = VectorFunctionSpace(mesh_N, op.space1, op.degree1) * FunctionSpace(mesh_N, op.space2, op.degree2)

# Specify physical and solver parameters
dt = adap.adaptTimestepSW(mesh, b)
Dt = Constant(dt)
T = op.Tend
Ts = op.Tstart
ndump = op.ndump
op.checkCFL(b)

# Get initial gauge values
gaugeData = {}
gauges = ("P02", "P06")
v0 = {}
for gauge in gauges:
    v0[gauge] = float(eta0.at(op.gaugeCoord(gauge)))

# Define Functions relating to goalBased approach
if useAdjoint:
    V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
    rho = Function(V_N)
    rho_u, rho_e = rho.split()
    rho_u.rename("Velocity residual")
    rho_e.rename("Elevation residual")
    dual = Function(V)
    dual_u, dual_e = dual.split()
    dual_N = Function(V_N)
    dual_N_u, dual_N_e = dual_N.split()
    dual_N_u.rename("Adjoint velocity")
    dual_N_e.rename("Adjoint elevation")
    P0_N = FunctionSpace(mesh_N, "DG", 0)
    v = TestFunction(P0_N)
    epsilon = Function(P0_N, name="Error indicator")

# Get adaptivity parameters
hmin = op.hmin
hmax = op.hmax
rm = op.rm
iStart = int(op.Tstart / dt)
iEnd = int(np.ceil(T / dt))
nEle, nVer = msh.meshStats(mesh)
mM = [nEle, nEle]           # Min/max #Elements
Sn = nEle
nVerT = nVer * op.vscale    # Target #Vertices
nVerT0 = nVerT

# Initialise counters
cnt = 0
endT = 0.

if getData:
    # Get solver parameter values and construct solver
    primalTimer = clock()
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = False
    options.use_grad_depth_viscosity_term = False
    options.simulation_export_time = dt * ndump
    options.simulation_end_time = T
    options.timestepper_type = op.timestepper
    options.timestep = dt
    options.output_directory = dirName
    options.export_diagnostics = True
    options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']

    # Apply ICs and time integrate
    solver_obj.assign_initial_conditions(elev=eta0)
    solver_obj.iterate()
    primalTimer = clock()-primalTimer
    print('Time elapsed for fixed mesh solver: %.1fs (%.2fmins)' % (primalTimer, primalTimer / 60))

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
    if useAdjoint & op.gradate:
        h0 = Function(FunctionSpace(mesh, "CG", 1)).interpolate(CellSize(mesh))
    mn = 0
    adaptTimer = clock()
    while mn < np.ceil(T / (dt * rm)):
        stepTimer = clock()
        index = mn * int(rm / ndump)
        V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)

        # Enforce ICs / load variables from disk
        uv_2d = Function(V.sub(0))
        elev_2d = Function(V.sub(1))
        if index == 0:
            elev_2d.interpolate(eta0)
            uv_2d.interpolate(Expression((0, 0)))
        else:
            indexStr = msc.indexString(index)
            with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            with DumbCheckpoint(dirName + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                loadVel.load(uv_2d, name='uv_2d')
                loadVel.close()

        # Construct metric
        W = TensorFunctionSpace(mesh, 'CG', 1)
        if useAdjoint & (mn != 0):
            # TODO properly (see above). Test in this form
            # Load error indicator data from HDF5 and interpolate onto a P1 space defined on current mesh
            # with DumbCheckpoint(dirName + 'hdf5/error_' + msc.indexString(cnt), mode=FILE_READ) as loadError:
            with DumbCheckpoint('plots/firedrake-tsunami/hdf5/error_' + msc.indexString(mn), mode=FILE_READ) as loadError:
                loadError.load(epsilon)
                loadError.close()
            errEst = Function(FunctionSpace(mesh, "CG", 1)).interpolate(inte.interp(mesh, epsilon)[0])
            M = adap.isotropicMetric(W, errEst, op=op, invert=False)
        else:
            if op.mtype != 's':
                if op.iso:
                    M = adap.isotropicMetric(W, elev_2d, op=op)
                else:
                    H = adap.constructHessian(mesh, W, elev_2d, op=op)
                    M = adap.computeSteadyMetric(mesh, W, H, elev_2d, nVerT=nVerT, op=op)
            if op.mtype != 'f':
                spd = Function(W.sub(1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                if op.iso:
                    M2 = adap.isotropicMetric(W, spd, op=op)
                else:
                    H = adap.constructHessian(mesh, W, spd, op=op)
                    M2 = adap.computeSteadyMetric(mesh, W, H, spd, nVerT=nVerT, op=op)
                M = adap.metricIntersection(mesh, W, M, M2) if op.mtype == 'b' else M2
        if op.gradate:
            if useAdjoint:
                M_ = adap.isotropicMetric(W, inte.interp(mesh, h0)[0], bdy=True, op=op)  # Initial boundary metric
                M = adap.metricIntersection(mesh, W, M, M_, bdy=True)
            adap.metricGradation(mesh, M)
            # TODO: always gradate to coast
        if op.advect:
            M = adap.advectMetric(M, uv_2d, 2*Dt, n=3*rm)
            # TODO: isotropic advection?

        # Adapt mesh and interpolate variables
        mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
        V = VectorFunctionSpace(mesh, op.space1, op.degree1) * FunctionSpace(mesh, op.space2, op.degree2)
        elev_2d, uv_2d, b = inte.interp(mesh, elev_2d, uv_2d, b)
        uv_2d.rename('uv_2d')
        elev_2d.rename('elev_2d')

        # Adapt timestep
        if tAdapt:
            dt = adap.adaptTimestepSW(mesh, b)
            Dt.assign(dt)

        # Establish Thetis flow solver object
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = False
        options.use_grad_depth_viscosity_term = False
        options.simulation_export_time = dt * ndump
        startT = endT
        endT += dt * rm
        options.simulation_end_time = endT
        options.timestepper_type = op.timestepper
        options.timestep = dt
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

        # Timestepper bookkeeping for export time step and next export
        solver_obj.i_export = index
        solver_obj.iteration = mn * rm
        solver_obj.simulation_time = startT
        solver_obj.next_export_t = startT + options.simulation_export_time  # For next export
        for e in solver_obj.exporters.values():
            e.set_next_export_ix(solver_obj.i_export)

        # Time integrate and print
        solver_obj.iterate()
        nEle = msh.meshStats(mesh)[0]
        mM = [min(nEle, mM[0]), max(nEle, mM[1])]
        Sn += nEle
        mn += 1
        op.printToScreen(mn, clock() - adaptTimer, clock() - stepTimer, nEle, Sn, mM, endT)

    # Extract gauge timeseries data
    if op.gauges:
            gaugeData = tim.extractTimeseries(gauges, elev_2d, gaugeData, v0, op=op)
    adaptTimer = clock() - adaptTimer
    print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (adaptTimer, adaptTimer / 60))

# Print to screen timing analyses
if getData and useAdjoint:
    msc.printTimings(primalTimer, dualTimer, errorTimer, adaptTimer)

# Save and plot timeseries
name = input("Enter a name for these time series (e.g. 'goalBased8-12-17'): ") or 'test'
for gauge in gauges:
    tim.saveTimeseries(gauge, gaugeData[gauge], name=name)
    tim.plotGauges(gauge, int(T), op=op)