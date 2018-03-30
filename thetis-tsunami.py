from thetis_adjoint import *
from thetis.field_defs import field_metadata
from firedrake_adjoint import *
import pyadjoint
from fenics_adjoint.solving import SolveBlock

import numpy as np
from time import clock
import datetime

from utils.adaptivity import isoP2, constructGradient, isotropicMetric, steadyMetric, metricIntersection, metricGradation
from utils.callbacks import TohokuCallback, ShallowWaterCallback, RossbyWaveCallback
from utils.callbacks import ObjectiveTohokuCallback, ObjectiveSWCallback, ObjectiveRWCallback
from utils.error import explicitErrorEstimator, fluxJumpError
from utils.forms import formsSW, interelementTerm, strongResidualSW
from utils.interpolation import *
from utils.mesh import TohokuDomain, domainSW, domainRW, meshStats
from utils.misc import cheatCodes, indexString, getMax
from utils.options import *

now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)


def solverSW(startRes, approach, getData, getError, useAdjoint, aposteriori, mode='tohoku', op=Options()):
    """
    Run mesh adaptive simulations for the Tohoku problem.

    :param startRes: Starting resolution.
    :param approach: meshing method.
    :param getData: run forward simulation?
    :param getError: generate error estimates?
    :param useAdjoint: run adjoint simulation?
    :param aposteriori: error estimator classification.
    :param mode: test case or main script used.
    :param op: parameter values.
    :return: mean element count and relative error in objective functional value.
    """
    fullTime = clock()
    try:
        assert(mode in ('tohoku', 'shallow-water', 'rossby-wave'))
    except:
        raise NotImplementedError
    try:
        assert (float(physical_constants['g_grav'].dat.data) == op.g)
    except:
        physical_constants['g_grav'].assign(op.g)
    primalTimer = dualTimer = errorTimer = adaptTimer = False

    # Establish filenames
    di = 'plots/'+mode+'/'
    if op.plotpvd:
        residualFile = File(di + "residual.pvd")
        implicitErrorFile = File(di + "implicitError.pvd")
        errorFile = File(di + "errorIndicator.pvd")
        adjointFile = File(di + "adjoint.pvd")

    # Load Mesh, initial condition and bathymetry
    if mode == 'tohoku':
        mesh_H, eta0, b, BCs, f = TohokuDomain(startRes, wd=op.wd)
    elif mode == 'shallow-water':
        mesh_H, eta0, b, BCs, f = domainSW(startRes)
    else:
        mesh_H, u0, eta0, b, BCs, f = domainRW(startRes, op=op)

    # Define initial FunctionSpace and variables of problem and apply initial conditions
    V = op.mixedSpace(mesh_H)
    q = Function(V)
    uv_2d, elev_2d = q.split()  # Needed to load data into
    uv_2d.rename("uv_2d")
    elev_2d.rename("elev_2d")
    P1 = FunctionSpace(mesh_H, "CG", 1)
    if approach in ('residual', 'implicit', 'DWR', 'DWE', 'explicit'):
        q_ = Function(V)        # Variable at previous timestep
        uv_2d_, elev_2d_ = q_.split()
    P0 = FunctionSpace(mesh_H, "DG", 0)

    # Define Functions relating to a posteriori estimators
    if aposteriori or approach == 'norm':
        if useAdjoint:                          # Define adjoint variables
            dual = Function(V)
            dual_u, dual_e = dual.split()
            dual_u.rename("Adjoint velocity")
            dual_e.rename("Adjoint elevation")
        if op.orderChange:                      # Define variables on higher/lower order space
            Ve = op.mixedSpace(mesh_H, orderChange=op.orderChange)
            qe = Function(Ve)
            ue, ee = qe.split()
            qe_ = Function(Ve)
            ue_, ee_ = qe_.split()
            if useAdjoint:
                duale = Function(Ve)
                duale_u, duale_e = duale.split()
            be = Function(FunctionSpace(mesh_H, "CG", 1+op.orderChange)).assign(float(np.average(b.dat.data)))
        elif op.refinedSpace:                   # Define variables on refined space
            mesh_h = isoP2(mesh_H)
            Ve = op.mixedSpace(mesh_h)
            if mode == 'tohoku':
                be = TohokuDomain(mesh=mesh_h)[2]
            else:
                be = Function(FunctionSpace(mesh_h, "CG", 1)).assign(float(np.average(b.dat.data)))
            qe = Function(Ve)
            P0 = FunctionSpace(mesh_h, "DG", 0)
        else:                                   # Copy standard variables to mimic enriched space labels
            Ve = V
            qe = q
            qe_ = q_
            be = b
        if approach in ('implicit', 'DWE'):     # Define variables for implicit error estimation
            e_ = Function(Ve)
            e = Function(Ve)
            e0, e1 = e.split()
            e0.rename("Implicit error 0")
            e1.rename("Implicit error 1")
            et = TestFunction(Ve)
            (et0, et1) = (as_vector((et[0], et[1])), et[2])
            normal = FacetNormal(mesh_H)
        elif approach in ('residual', 'explicit', 'DWR'):
            rho = Function(Ve)
            rho_u, rho_e = rho.split()
            rho_u.rename("Velocity residual")
            rho_e.rename("Elevation residual")
    v = TestFunction(P0)

    # Initialise parameters and counters
    nEle, nVerT = meshStats(mesh_H)
    nVerT *= op.vscale                      # Target #Vertices
    mM = [nEle, nEle]                       # Min/max #Elements
    Sn = nEle
    endT = 0.
    dt = op.dt
    Dt = Constant(dt)
    T = op.Tend
    iEnd = int(T / (dt * op.rm))

    # Get initial boundary metric and TODO wetting and drying parameter
    if op.gradate or op.wd:
        H0 = Function(P1).interpolate(CellSize(mesh_H))
    if op.wd:
        g = constructGradient(elev_2d)
        spd = assemble(v * sqrt(inner(g, g)) * dx)
        gs = np.min(np.abs(spd.dat.data))
        print('#### gradient = ', gs)
        ls = np.min([H0.dat.data[i] for i in DirichletBC(P1, 0, 'on_boundary').nodes])
        print('#### ls = ', ls)
        alpha = Constant(gs * ls)
        print('#### alpha = ', alpha.dat.data)
        # alpha = Constant(0.5)
        # exit(23)

    tic = clock()
    if getData:
        if op.printStats:
            print('Starting fixed mesh primal run (forwards in time)')

        # Get solver parameter values and construct solver
        solver_obj = solver2d.FlowSolver2d(mesh_H, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True if op.nonlinear else False
        options.use_grad_depth_viscosity_term = False
        options.use_grad_div_viscosity_term = False
        if mode == 'rossby-wave':
            options.coriolis_frequency = f
        options.simulation_export_time = dt * (op.rm-1) if aposteriori else dt * op.ndump
        options.simulation_end_time = T
        options.period_of_interest_start = op.Tstart
        options.period_of_interest_end = T
        options.timestepper_type = op.timestepper
        options.timestep = dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.log_output = op.printStats
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        options.use_wetting_and_drying = op.wd
        if op.wd:
            options.wetting_and_drying_alpha = alpha
        if mode == 'rossby-wave':
            solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        else:
            solver_obj.assign_initial_conditions(elev=eta0)
        if mode == 'tohoku':
            cb1 = TohokuCallback(solver_obj)
            cb2 = ObjectiveTohokuCallback(solver_obj)
        elif mode == 'shallow-water':
            cb1 = ShallowWaterCallback(solver_obj)
            cb2 = ObjectiveSWCallback(solver_obj)
        else:
            cb1 = RossbyWaveCallback(solver_obj)
            cb2 = ObjectiveRWCallback(solver_obj)
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.add_callback(cb2, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs
        if aposteriori and approach != 'DWF':
            if mode == 'tohoku':
                def selector():
                    t = solver_obj.simulation_time
                    rm = options.timesteps_per_remesh
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            elif mode == 'shallow-water':
                def selector():
                    t = solver_obj.simulation_time
                    rm = options.timesteps_per_remesh
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            else:
                def selector():
                    t = solver_obj.simulation_time
                    rm = options.timesteps_per_remesh
                    dt = options.timestep
                    options.simulation_export_time = dt if int(t / dt) % rm == 0 else (rm - 1) * dt
            primalTimer = clock()
            solver_obj.iterate(export_func=selector)
        else:
            primalTimer = clock()
            solver_obj.iterate()
        primalTimer = clock() - primalTimer
        J_h = cb1.__call__()[1]    # Evaluate objective functional
        if op.printStats:
            print('Primal run complete. Run time: %.3fs' % primalTimer)

        # Reset counters
        cntT = int(np.ceil(T/dt))
        if useAdjoint:
            if op.printStats:
                print('\nGenerating dual solutions...')

            # Assemble objective functional
            Jfuncs = cb2.__call__()[1]
            J = 0
            for i in range(1, len(Jfuncs)):
                J += 0.5 * (Jfuncs[i - 1] + Jfuncs[i]) * dt

            # Compute gradient
            dualTimer = clock()
            dJdb = compute_gradient(J, Control(b))      # TODO: Rewrite pyadjoint code to avoid computing this
            File(di + 'gradient.pvd').write(dJdb)
            print("Norm of gradient = %e" % dJdb.dat.norm)

            # Extract adjoint solutions
            tape = get_working_tape()
            # tape.visualise()
            solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
            N = len(solve_blocks)
            r = N % op.rm   # Number of extra tape annotations in setup
            for i in range(N-1, r-2, -op.rm):
                dual.assign(solve_blocks[i].adj_sol)
                dual_u, dual_e = dual.split()
                dual_u.rename('Adjoint velocity')
                dual_e.rename('Adjoint elevation')
                with DumbCheckpoint(di+'hdf5/adjoint_'+indexString(int((i-r+1)/op.rm)), mode=FILE_CREATE) as saveAdj:
                    saveAdj.store(dual_u)
                    saveAdj.store(dual_e)
                    saveAdj.close()
                if op.plotpvd:
                    adjointFile.write(dual_u, dual_e, time=dt * (i-r+1))
                if op.printStats:
                    print('Adjoint simulation %.2f%% complete' % ((N - i + r - 1) / N * 100))
            dualTimer = clock() - dualTimer
            if op.printStats:
                print('Dual run complete. Run time: %.3fs' % dualTimer)
    cnt = 0

    # Loop back over times to generate error estimators
    if getError:
        if op.printStats:
            print('\nStarting error estimate generation')
        errorTimer = clock()

        # Define implicit error problem
        if approach in ('implicit', 'DWE'): # TODO: Check out situation with nonlinear option
            B_, L = formsSW(qe, qe_, b, Dt, impermeable=False, coriolisFreq=f, op=op)
            B = formsSW(e, e_, b, Dt, impermeable=False, coriolisFreq=f, op=op)[0]
            I = interelementTerm(et1 * ue, n=normal) * dS
            errorProblem = NonlinearVariationalProblem(B - L + B_ - I, e)
            errorSolver = NonlinearVariationalSolver(errorProblem, solver_parameters=op.params)

        for k in range(0, iEnd):
            if op.printStats:
                print('Generating error estimate %d / %d' % (k+1, iEnd))

            if approach == 'DWF':
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(k), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(k), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
            else:
                i1 = 0 if k == 0 else 2*k-1
                i2 = 2*k
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(i1), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(i1), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()
                uv_2d_.assign(uv_2d)
                elev_2d_.assign(elev_2d)
                with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(i2), mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d)
                    loadVel.close()
                with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(i2), mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d)
                    loadElev.close()

            # Solve implicit error problem
            if approach in ('implicit', 'DWE') or op.orderChange:
                ue.interpolate(uv_2d)
                ee.interpolate(elev_2d)
                ue_.interpolate(uv_2d_)
                ee_.interpolate(elev_2d_)
                if approach in ('implicit', 'DWE'):
                    errorSolver.solve()
                    e_.assign(e)
                    if approach == 'implicit':
                        epsilon = assemble(v * sqrt(inner(e, e)) * dx)
                    if op.plotpvd:
                        implicitErrorFile.write(e0, e1, time=float(k))

            # Approximate residuals
            if approach in ('explicit', 'residual', 'DWR'):
                if op.refinedSpace:
                    qe, qe_ = mixedPairInterp(mesh_h, Ve, q, q_)
                Au, Ae = strongResidualSW(qe, qe_, be, Dt, coriolisFreq=None, op=op)
                rho_u.interpolate(Au)
                rho_e.interpolate(Ae)
                if op.plotpvd:
                    residualFile.write(rho_u, rho_e, time=float(k))
                if approach == 'residual':
                    epsilon = assemble(v * sqrt(inner(rho, rho)) * dx)
                elif approach == 'explicit':
                    epsilon = explicitErrorEstimator(qe, rho, be if op.orderChange else be, v, maxBathy=True)

            if useAdjoint:
                with DumbCheckpoint(di+'hdf5/adjoint_'+indexString(k), mode=FILE_READ) as loadAdj:
                    loadAdj.load(dual_u)
                    loadAdj.load(dual_e)
                    loadAdj.close()
                if approach == 'DWR':
                    if op.orderChange:
                        duale_u.interpolate(dual_u)
                        duale_e.interpolate(dual_e)
                        epsilon = assemble(v * inner(rho, duale) * dx)
                    elif op.refinedSpace:
                        duale = mixedPairInterp(mesh_h, dual)[0]
                        epsilon = assemble(v * inner(rho, duale) * dx)
                    else:
                        epsilon = assemble(v * inner(rho, dual) * dx)
                elif approach == 'DWE':
                    if op.orderChange:
                        duale_u.interpolate(dual_u)
                        duale_e.interpolate(dual_e)
                        epsilon = assemble(v * inner(e, duale) * dx)
                    elif op.refinedSpace:
                        duale = mixedPairInterp(mesh_h, dual)[0]
                        epsilon = assemble(v * inner(e, duale) * dx)
                    else:
                        epsilon = assemble(v * inner(e, dual) * dx)
                elif approach == 'DWF':
                    raise NotImplementedError   # TODO: maximise DWF over time window

            # Store error estimates
            epsilon.rename("Error indicator")
            with DumbCheckpoint(di+'hdf5/'+approach+'Error'+indexString(k), mode=FILE_CREATE) as saveErr:
                saveErr.store(epsilon)
                saveErr.close()
            if op.plotpvd:
                errorFile.write(epsilon, time=float(k))
        errorTimer = clock() - errorTimer
        if op.printStats:
            print('Errors estimated. Run time: %.3fs' % errorTimer)

    if approach != 'fixedMesh':
        with pyadjoint.stop_annotating():
            q = Function(V)
            uv_2d, elev_2d = q.split()
            elev_2d.interpolate(eta0)
            if aposteriori:
                epsilon = Function(P0, name="Error indicator")
            if op.printStats:
                print('\nStarting adaptive mesh primal run (forwards in time)')
            adaptTimer = clock()
            # while cnt < np.ceil(T / dt):
            while cnt < int(T / dt):        # It appears this format is better for CFL criterion derived timesteps
                stepTimer = clock()

                # Load variables from disk
                if cnt != 0:
                    V = op.mixedSpace(mesh_H)
                    q = Function(V)
                    uv_2d, elev_2d = q.split()
                    with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(int(cnt/op.ndump)), mode=FILE_READ) \
                            as loadElev:
                        loadElev.load(elev_2d, name='elev_2d')
                        loadElev.close()
                    with DumbCheckpoint(di+'hdf5/Velocity2d_'+indexString(int(cnt/op.ndump)), mode=FILE_READ) \
                            as loadVel:
                        loadVel.load(uv_2d, name='uv_2d')
                        loadVel.close()

                # Construct metric
                if aposteriori:
                    with DumbCheckpoint(di+'hdf5/'+approach+'Error'+indexString(int(cnt/op.rm)), mode=FILE_READ) \
                            as loadErr:
                        loadErr.load(epsilon)
                        loadErr.close()
                    errEst = Function(FunctionSpace(mesh_H, "CG", 1)).interpolate(interp(mesh_H, epsilon)[0])
                    M = isotropicMetric(errEst, op=op, invert=False, nVerT=nVerT)
                else:
                    if approach == 'norm':
                        v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                        epsilon = assemble(v * inner(q, q) * dx)
                        M = isotropicMetric(epsilon, invert=False, nVerT=nVerT, op=op)
                    elif approach =='fluxJump' and cnt != 0:
                        v = TestFunction(FunctionSpace(mesh_H, "DG", 0))
                        epsilon = fluxJumpError(q, v)
                        M = isotropicMetric(epsilon, invert=False, nVerT=nVerT, op=op)
                    else:
                        if op.mtype != 's':
                            if approach == 'fieldBased':
                                M = isotropicMetric(elev_2d, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'gradientBased':
                                g = constructGradient(elev_2d)
                                M = isotropicMetric(g, invert=False, nVerT=nVerT, op=op)
                            elif approach == 'hessianBased':
                                M = steadyMetric(elev_2d, nVerT=nVerT, op=op)
                        if cnt != 0:    # Can't adapt to zero velocity
                            if op.mtype != 'f':
                                spd = Function(FunctionSpace(mesh_H, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                                if approach == 'fieldBased':
                                    M2 = isotropicMetric(spd, invert=False, nVerT=nVerT, op=op)
                                elif approach == 'gradientBased':
                                    g = constructGradient(spd)
                                    M2 = isotropicMetric(g, invert=False, nVerT=nVerT, op=op)
                                elif approach == 'hessianBased':
                                    M2 = steadyMetric(spd, nVerT=nVerT, op=op)
                                M = metricIntersection(M, M2) if op.mtype == 'b' else M2
                if op.gradate:
                    M_ = isotropicMetric(interp(mesh_H, H0)[0], bdy=True, op=op)  # Initial boundary metric
                    M = metricIntersection(M, M_, bdy=True)
                    metricGradation(M, op=op)
                if op.plotpvd:
                    File('plots/'+mode+'/mesh.pvd').write(mesh_H.coordinates, time=float(cnt))

                # Adapt mesh and interpolate variables
                if not (((approach in ('fieldBased', 'gradientBased', 'hessianBased') and op.mtype != 'f')
                         or approach == 'fluxJump') and cnt == 0):
                    mesh_H = AnisotropicAdaptation(mesh_H, M).adapted_mesh
                    P1 = FunctionSpace(mesh_H, "CG", 1)
                    elev_2d, uv_2d = interp(mesh_H, elev_2d, uv_2d)
                    if mode == 'tohoku':
                        b = interp(mesh_H, b)[0]
                    elif mode == 'shallow-water':
                        b = Function(P1).assign(0.1)
                    else:
                        b = Function(P1).assign(1.)
                    uv_2d.rename('uv_2d')
                    elev_2d.rename('elev_2d')

                # Solver object and equations
                adapSolver = solver2d.FlowSolver2d(mesh_H, b)
                adapOpt = adapSolver.options
                adapOpt.element_family = op.family
                adapOpt.use_nonlinear_equations = True if op.nonlinear else False
                adapOpt.use_grad_depth_viscosity_term = False
                adapOpt.use_grad_div_viscosity_term = False
                adapOpt.simulation_export_time = dt * op.ndump
                startT = endT
                endT += dt * op.rm
                adapOpt.simulation_end_time = endT
                adapOpt.period_of_interest_start = op.Tstart
                adapOpt.period_of_interest_end = T
                adapOpt.timestepper_type = op.timestepper
                adapOpt.timestep = dt
                adapOpt.output_directory = di
                adapOpt.log_output = op.printStats
                adapOpt.export_diagnostics = True
                adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
                adapOpt.use_wetting_and_drying = op.wd
                if op.wd:
                    adapOpt.wetting_and_drying_alpha = alpha
                if mode == 'rossby-wave':
                    adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh_H)[1])
                field_dict = {'elev_2d': elev_2d, 'uv_2d': uv_2d}
                e = exporter.ExportManager(di + 'hdf5',
                                           ['elev_2d', 'uv_2d'],
                                           field_dict,
                                           field_metadata,
                                           export_type='hdf5')
                adapSolver.assign_initial_conditions(elev=elev_2d, uv=uv_2d)
                adapSolver.i_export = int(cnt / op.ndump)
                adapSolver.iteration = cnt
                adapSolver.simulation_time = startT
                adapSolver.next_export_t = startT + adapOpt.simulation_export_time  # For next export
                for e in adapSolver.exporters.values():
                    e.set_next_export_ix(adapSolver.i_export)

                # Evaluate callbacks and iterate
                if mode == 'tohoku':
                    cb1 = TohokuCallback(adapSolver)
                elif mode == 'shallow-water':
                    cb1 = ShallowWaterCallback(adapSolver)
                elif mode == 'rossby-wave':
                    cb1 = RossbyWaveCallback(adapSolver)
                if cnt != 0:
                    cb1.objective_value = J_h
                adapSolver.add_callback(cb1, 'timestep')
                solver_obj.bnd_functions['shallow_water'] = BCs
                adapSolver.iterate()
                J_h = cb1.__call__()[1]  # Evaluate objective functional

                # Get mesh stats
                nEle = meshStats(mesh_H)[0]
                mM = [min(nEle, mM[0]), max(nEle, mM[1])]
                Sn += nEle
                cnt += op.rm
                av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt*dt, dt)

            adaptTimer = clock() - adaptTimer
            if op.printStats:
                print('Adaptive primal run complete. Run time: %.3fs' % adaptTimer)
    else:
        av = nEle
    fullTime = clock() - fullTime
    if op.printStats:
        printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, fullTime)

    # Measure error using metrics, using data from Huang et al.
    if mode == 'rossby-wave':   # TODO: Plot / interpret these results
        index = int(cntT/op.ndump) if approach == 'fixedMesh' else int((cnt-op.rm) / op.ndump)
        with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(index), mode=FILE_READ) as loadElev:
            loadElev.load(elev_2d, name='elev_2d')
            loadElev.close()
        # peak_i, peak = getMax(interp(isoP2(mesh_H), elev_2d)[0].dat.data)
        peak_i, peak = getMax(elev_2d.dat.data)
        dgCoords = Function(VectorFunctionSpace(mesh_H, op.space2, op.degree2)).interpolate(mesh_H.coordinates)
        distanceTravelled = np.abs(dgCoords.dat.data[peak_i][0])

    toc = clock() - tic
    if mode == 'rossby-wave':
        return av, np.abs(peak/0.1567020), distanceTravelled, distanceTravelled/47.18, toc
    else:
        return av, np.abs(op.J(mode) - J_h)/np.abs(op.J(mode)), J_h, toc

    # TODO: Also generate and output a timeseries plot for the integrand of the objective functional [Anca Belme paper]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}")
    parser.add_argument("approach", help="Choose error estimator from {'norm', 'fieldBased', 'gradientBased', "
                                         "'hessianBased', 'residual', 'explicit', 'fluxJump', 'implicit', 'DWF', "
                                         "'DWR', 'DWE'}" )
    parser.add_argument("-w", help="Use wetting and drying")
    parser.add_argument("-b", help="Use bootstrapping")
    parser.add_argument("-ho", help="Compute errors and residuals in a higher order space")
    parser.add_argument("-lo", help="Compute errors and residuals in a lower order space")
    parser.add_argument("-r", help="Compute errors and residuals in a refined space")
    args = parser.parse_args()
    orderBool = (not args.ho) or (not args.lo)
    assert(orderBool)
    assert(orderBool or (not args.r))
    print("Mode: ", args.mode)
    print("Approach: ", args.approach)
    mode = args.mode

    # Choose mode and set parameter values
    approach, getData, getError, useAdjoint, aposteriori = cheatCodes(args.approach)
    op = Options(mode=mode,
                 vscale=0.85,
                 family='dg-dg',
                 rm=100 if useAdjoint else 50,
                 gradate=True if aposteriori else False,
                 window=True if approach == 'DWF' else False,   # TODO
                 outputMetric=False,
                 plotpvd=True,
                 gauges=False,  # TODO: Include callbacks for Tohoku case
                 bootstrap=True if args.b else False,
                 printStats=False,
                 wd=True if args.w else False,
                 ndump=50)
    if mode == 'shallow-water':
        op.rm = 10 if useAdjoint else 5
    elif mode == 'rossby-wave':
        op.rm = 48 if useAdjoint else 24
    if args.ho:
        op.orderChange = 1
    elif args.lo:
        op.orderChange = -1
    elif args.r:
        op.refinedSpace = True

    # Run simulation(s)
    filename = 'outdata/outputs/'+mode+'/'+approach+date
    if op.bootstrap:
        filename += '_BOOTSTRAP'
    textfile = open(filename +'.txt', 'w+')
    if op.bootstrap:
        # for i in range(11):   # TODO: Can't currently do multiple adjoint runs
        for i in range(5, 6):
            av, rel, J_h, timing = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
            var = np.abs(J_h - J_h_) if i > 0 else 0.
            J_h_ = J_h
            print('Run %d:  Mean element count %6d      Objective value %.4e        Timing %.1fs    Difference %.4e'
                  % (i, av, J_h, timing, var))
            textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, J_h, timing, var))
    else:
        for i in range(1, 6):
            if mode == 'rossby-wave':
                av, relativePeak, distanceTravelled, phaseSpd, timing = \
                    solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
                print('Run %d: <#Elements>: %6d  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm  Timing %.1fs'
                      % (i, av, relativePeak, distanceTravelled, phaseSpd, timing))
                textfile.write('%d, %.4f, %.4f, %.4f, %.1f\n' % (av, relativePeak, distanceTravelled, phaseSpd, timing))
            else:
                av, rel, J_h, tim = solverSW(i, approach, getData, getError, useAdjoint, aposteriori, mode=mode, op=op)
                print('Run %d: Mean element count %6d Relative error %.4e Timing %.1fs'
                      % (i, av, rel, tim))
                textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, tim, J_h))
    textfile.close()
