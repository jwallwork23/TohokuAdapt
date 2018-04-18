from thetis_adjoint import *
import pyadjoint

import numpy as np
from time import clock

from utils.adaptivity import metricIntersection, steadyMetric
from utils.callbacks import *
from utils.forms import solutionRW
from utils.interpolation import interp
from utils.mesh import meshStats, problemDomain
from utils.misc import indexString, peakAndDistance
from utils.options import Options


def fixedMesh(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/fixedMesh/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        nEle = meshStats(mesh)
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        if op.mode == 'rossby-wave':
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state

        # Initialise solver
        solver_obj = solver2d.FlowSolver2d(mesh, b)
        options = solver_obj.options
        options.element_family = op.family
        options.use_nonlinear_equations = True if op.nonlinear else False
        options.use_grad_depth_viscosity_term = False
        options.use_grad_div_viscosity_term = False
        options.use_lax_friedrichs_velocity = False  # TODO: This is a temporary fix
        if op.mode == 'rossby-wave':
            options.coriolis_frequency = f
        options.simulation_export_time = op.dt * op.ndump
        options.simulation_end_time = op.Tend
        options.period_of_interest_start = op.Tstart
        options.period_of_interest_end = op.Tend
        options.timestepper_type = op.timestepper
        options.timestep = op.dt
        options.timesteps_per_remesh = op.rm
        options.output_directory = di
        options.export_diagnostics = True
        options.log_output = op.printStats
        options.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
        options.use_wetting_and_drying = op.wd
        # if op.wd:                                         TODO
        #     options.wetting_and_drying_alpha = alpha
        solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
        if op.mode == 'rossby-wave':
            cb1 = RossbyWaveCallback(solver_obj)
            cb2 = ObjectiveRWCallback(solver_obj)
        elif op.mode == 'shallow-water':
            cb1 = ShallowWaterCallback(solver_obj)
            cb2 = ObjectiveSWCallback(solver_obj)
        else:
            cb1 = TohokuCallback(solver_obj)
            cb2 = ObjectiveTohokuCallback(solver_obj)
            cb3 = P02Callback(solver_obj)
            cb4 = P06Callback(solver_obj)
            solver_obj.add_callback(cb3, 'timestep')
            solver_obj.add_callback(cb4, 'timestep')
        solver_obj.add_callback(cb1, 'timestep')
        solver_obj.add_callback(cb2, 'timestep')
        solver_obj.bnd_functions['shallow_water'] = BCs

        # Solve and extract timeseries / functionals
        primalTimer = clock()
        solver_obj.iterate()
        primalTimer = clock() - primalTimer
        J_h = cb1.quadrature()          # Evaluate objective functional
        integrand = cb1.__call__()[1]   # and get integrand timeseries
        if op.mode == 'tohoku':
            totalVarP02 = cb3.totalVariation()
            totalVarP06 = cb4.totalVariation()
        if op.printStats:
            print('Primal run complete. Run time: %.3fs' % primalTimer)

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT/op.ndump)
            with DumbCheckpoint(di+'hdf5/Elevation2d_'+indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return nEle, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, primalTimer
        elif op.mode == 'tohoku':
            return nEle, rel, J_h, integrand, totalVarP02, totalVarP06, primalTimer
        else:
            return nEle, rel, J_h, integrand, primalTimer


def hessianBased(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/fixedMesh/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        V = op.mixedSpace(mesh)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into
        if op.mode == 'rossby-wave':
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state
        elev_2d.interpolate(eta0)
        uv_2d.interpolate(u0)

        # Initialise parameters and counters
        nEle, op.nVerT = meshStats(mesh)
        op.nVerT *= op.rescaling  # Target #Vertices
        mM = [nEle, nEle]  # Min/max #Elements
        Sn = nEle
        cnt = 0
        endT = 0.

        adaptTimer = clock()
        while cnt < op.cntT:
            stepTimer = clock()
            indexStr = indexString(int(cnt / op.ndump))

            # Load variables from disk
            if cnt != 0:
                V = op.mixedSpace(mesh)
                q = Function(V)
                uv_2d, elev_2d = q.split()
                with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
                    loadElev.load(elev_2d, name='elev_2d')
                    loadElev.close()
                with DumbCheckpoint(di + 'hdf5/Velocity2d_' + indexStr, mode=FILE_READ) as loadVel:
                    loadVel.load(uv_2d, name='uv_2d')
                    loadVel.close()

            for l in range(op.nAdapt):  # TODO: Test this functionality

                # Construct metric
                if op.adaptField != 's':
                    M = steadyMetric(elev_2d, op=op)
                if cnt != 0:  # Can't adapt to zero velocity
                    if op.adaptField != 'f':
                        spd = Function(FunctionSpace(mesh, "DG", 1)).interpolate(sqrt(dot(uv_2d, uv_2d)))
                        M2 = steadyMetric(spd, op=op)
                        M = metricIntersection(M, M2) if op.adaptField == 'b' else M2

                # Adapt mesh and interpolate variables
                if cnt != 0 or op.adaptField == 's':
                    mesh = AnisotropicAdaptation(mesh, M).adapted_mesh
                    P1 = FunctionSpace(mesh, "CG", 1)
                    elev_2d, uv_2d = interp(mesh, elev_2d, uv_2d)
                    if op.mode == 'tohoku':
                        b = interp(mesh, b)
                    elif op.mode == 'shallow-water':
                        b = Function(P1).assign(0.1)
                    else:
                        b = Function(P1).assign(1.)
                    uv_2d.rename('uv_2d')
                    elev_2d.rename('elev_2d')

            # Solver object and equations
            adapSolver = solver2d.FlowSolver2d(mesh, b)
            adapOpt = adapSolver.options
            adapOpt.element_family = op.family
            adapOpt.use_nonlinear_equations = True if op.nonlinear else False
            adapOpt.use_grad_depth_viscosity_term = False
            adapOpt.use_grad_div_viscosity_term = False
            adapOpt.use_lax_friedrichs_velocity = False  # TODO: This is a temporary fix
            adapOpt.simulation_export_time = op.dt * op.ndump
            startT = endT
            endT += op.dt * op.rm
            adapOpt.simulation_end_time = endT
            adapOpt.period_of_interest_start = op.Tstart
            adapOpt.period_of_interest_end = op.Tend
            adapOpt.timestepper_type = op.timestepper
            adapOpt.timestep = op.dt
            adapOpt.output_directory = di
            adapOpt.export_diagnostics = True
            adapOpt.log_output = op.printStats
            adapOpt.fields_to_export_hdf5 = ['elev_2d', 'uv_2d']
            adapOpt.use_wetting_and_drying = op.wd
            # if op.wd:                                             TODO
            #     adapOpt.wetting_and_drying_alpha = alpha
            if op.mode == 'rossby-wave':
                adapOpt.coriolis_frequency = Function(P1).interpolate(SpatialCoordinate(mesh)[1])
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
            if op.mode == 'rossby-wave':
                cb1 = RossbyWaveCallback(adapSolver)
            elif op.mode == 'shallow-water':
                cb1 = ShallowWaterCallback(adapSolver)
            else:
                cb1 = TohokuCallback(adapSolver)
                cb3 = P02Callback(adapSolver)
                cb4 = P06Callback(adapSolver)
            if cnt != 0:
                cb1.objective_value = integrand
                if op.mode == 'tohoku':
                    cb3.gauge_values = gP02
                    cb4.gauge_values = gP06
            adapSolver.add_callback(cb1, 'timestep')
            if op.mode == 'tohoku':
                adapSolver.add_callback(cb3, 'timestep')
                adapSolver.add_callback(cb4, 'timestep')
            adapSolver.bnd_functions['shallow_water'] = BCs
            adapSolver.iterate()
            J_h = cb1.quadrature()
            integrand = cb1.__call__()[1]
            if op.mode == 'tohoku':
                gP02 = cb3.__call__()[1]
                gP06 = cb4.__call__()[1]
                totalVarP02 = cb3.totalVariation()
                totalVarP06 = cb4.totalVariation()

            # Get mesh stats
            nEle = meshStats(mesh)[0]
            mM = [min(nEle, mM[0]), max(nEle, mM[1])]
            Sn += nEle
            cnt += op.rm
            av = op.printToScreen(int(cnt/op.rm+1), clock()-adaptTimer, clock()-stepTimer, nEle, Sn, mM, cnt * op.dt)
        adaptTimer = clock() - adaptTimer   # TODO: This is timing more than in fixedMesh case

        # Measure error using metrics, as in Huang et al.
        if op.mode == 'rossby-wave':
            index = int(op.cntT / op.ndump)
            with DumbCheckpoint(di + 'hdf5/Elevation2d_' + indexString(index), mode=FILE_READ) as loadElev:
                loadElev.load(elev_2d, name='elev_2d')
                loadElev.close()
            peak, distance = peakAndDistance(elev_2d, op=op)
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return av, rel, J_h, integrand, np.abs(peak / peak_a), distance, distance / distance_a, adaptTimer
        elif op.mode == 'tohoku':
            return av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, adaptTimer
        else:
            return av, rel, J_h, integrand, adaptTimer


def DWR(startRes, op=Options()):
    raise NotImplementedError


if __name__ == "__main__":
    import argparse
    import datetime


    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}")
    parser.add_argument("-approach", help="Choose adaptive approach from {'hessianBased', 'DWR'}")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-w", help="Use wetting and drying")
    args = parser.parse_args()
    mode = args.mode
    approach = args.mode
    if approach is None:
        approach = 'fixedMesh'
    else:
        assert approach in ('hessianBased', 'DWR')
    solver = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased, 'DWR': DWR}[approach]
    print("Mode: %s, approach: %s" % (mode, approach))

    # Choose mode and set parameter values
    op = Options(mode=mode,
                 # gradate=True if aposteriori and mode == 'tohoku' else False,
                 gradate=False,  # TODO: Fix this for tohoku case
                 plotpvd=True if args.o else False,
                 printStats=True,
                 wd=True if args.w else False)

    # Establish filename
    filename = 'outdata/' + mode + '/fixedMesh'
    if args.w:
        filename += '_w'
    filename += '_' + date
    errorfile = open(filename + '.txt', 'w+')
    integrandFile = open(filename + 'Integrand.txt', 'w+')

    # Run simulations
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    Jlist = np.zeros(len(resolutions))
    if mode == 'tohoku':
        g2list = np.zeros(len(resolutions))
        g6list = np.zeros(len(resolutions))
        gaugeFileP02 = open(filename + 'P02.txt', 'w+')
        gaugeFileP06 = open(filename + 'P06.txt', 'w+')
    for i in resolutions:
        # Get data and save to disk
        if mode == 'rossby-wave':
            av, rel, J_h, integrand, relativePeak, distance, phaseSpd, tim = solver(i, op=op)
            print("""Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs
        OF error: %.4e  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm"""
                  % (i, av, J_h, tim, rel, relativePeak, distance, phaseSpd))
            errorfile.write('%d, %.4e, %.4f, %.4f, %.4f, %.1f, %.4e\n'
                           % (av, rel, relativePeak, distance, phaseSpd, tim, J_h))
        elif mode == 'tohoku':
            av, rel, J_h, integrand, totalVarP02, totalVarP06, gP02, gP06, tim = solver(i, op=op)
            print("""Run %d: Mean element count: %6d Objective %.4e Timing %.1fs 
        OF error: %.4e P02: %.3f P06: %.3f""" % (i, av, J_h, tim, rel, totalVarP02, totalVarP06))
            errorfile.write('%d, %.4e, %.3f, %.3f, %.1f, %.4e\n' % (av, rel, totalVarP02, totalVarP06, tim, J_h))
            gaugeFileP02.writelines(["%s," % val for val in gP02])
            gaugeFileP02.write("\n")
            gaugeFileP06.writelines(["%s," % val for val in gP06])
            gaugeFileP06.write("\n")
        else:
            av, rel, J_h, integrand, tim = solver(i, op=op)
            print('Run %d: Mean element count: %6d Objective: %.4e OF error %.4e Timing %.1fs' % (i, av, J_h, rel, tim))
            errorfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, tim, J_h))
        integrandFile.writelines(["%s," % val for val in integrand])
        integrandFile.write("\n")

        # Calculate orders of convergence
        Jlist[i] = J_h
        if mode == 'tohoku':
            g2list[i] = totalVarP02
            g6list[i] = totalVarP06
        if i > 1:
            Jconv = (Jlist[i] - Jlist[i - 1]) / (Jlist[i - 1] - Jlist[i - 2])
            if mode == 'tohoku':
                g2conv = (g2list[i] - g2list[i - 1]) / (g2list[i - 1] - g2list[i - 2])
                g6conv = (g6list[i] - g6list[i - 1]) / (g6list[i - 1] - g6list[i - 2])
                print("Orders of convergence... J: %.4f, P02: %.4f, P06: %.4f" % (Jconv, g2conv, g6conv))
            else:
                print("Order of convergence: %.4f" % Jconv)
    errorfile.close()
    if mode == 'tohoku':
        gaugeFileP02.close()
        gaugeFileP06.close()
    integrandFile.close()
