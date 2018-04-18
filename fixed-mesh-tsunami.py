from thetis import *
import pyadjoint

import numpy as np
from time import clock

from utils.callbacks import *
from utils.forms import solutionRW
from utils.mesh import meshStats, problemDomain
from utils.misc import indexString, peakAndDistance
from utils.options import Options


def fixedMesh(startRes, op=Options()):
    with pyadjoint.stop_annotating():
        di = 'plots/' + op.mode + '/'

        # Initialise domain and physical parameters
        try:
            assert (float(physical_constants['g_grav'].dat.data) == op.g)
        except:
            physical_constants['g_grav'].assign(op.g)
        mesh_H, u0, eta0, b, BCs, f = problemDomain(startRes, op=op)
        nEls = meshStats(mesh_H)
        V = op.mixedSpace(mesh_H)
        uv_2d, elev_2d = Function(V).split()  # Needed to load data into

        # Initialise solver
        solver_obj = solver2d.FlowSolver2d(mesh_H, b)
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
            peak_a, distance_a = peakAndDistance(solutionRW(V, t=op.Tend).split()[1])  # Analytic final-time state
            print('Peak %.4f vs. %.4f, distance %.4f vs. %.4f' % (peak, peak_a, distance, distance_a))

        rel = np.abs(op.J - J_h) / np.abs(op.J)
        if op.mode == 'rossby-wave':
            return nEls, rel, J_h, integrand, np.abs(peak/peak_a), distance, distance/distance_a, primalTimer
        elif op.mode == 'tohoku':
            return nEls, rel, J_h, integrand, totalVarP02, totalVarP06, primalTimer
        else:
            return nEls, rel, J_h, integrand, primalTimer


if __name__ == "__main__":
    import argparse
    import datetime

    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-w", help="Use wetting and drying")
    args = parser.parse_args()
    mode = args.mode
    print("Mode: ", mode)

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
    textfile = open(filename + '.txt', 'w+')
    integrandFile = open(filename + 'Integrand.txt', 'w+')

    # Run simulations
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    Jlist = np.zeros(len(resolutions))
    if mode == 'tohoku':
        g2list = np.zeros(len(resolutions))
        g6list = np.zeros(len(resolutions))
    for i in resolutions:
        # Get data and save to disk
        if mode == 'rossby-wave':
            av, rel, J_h, integrand, relativePeak, distance, phaseSpd, tim = fixedMesh(i, op=op)
            print("""Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs
        OF error: %.4e  Height error: %.4f  Distance: %.4fm  Speed error: %.4fm"""
                  % (i, av, J_h, tim, rel, relativePeak, distance, phaseSpd))
            textfile.write('%d, %.4e, %.4f, %.4f, %.4f, %.1f, %.4e\n'
                           % (av, rel, relativePeak, distance, phaseSpd, tim, J_h))
        elif mode == 'tohoku':
            av, rel, J_h, integrand, totalVarP02, totalVarP06, tim = fixedMesh(i, op=op)
            print("""Run %d: Mean element count: %6d Objective %.4e Timing %.1fs 
        OF error: %.4e P02: %.3f P06: %.3f""" % (i, av, J_h, tim, rel, totalVarP02, totalVarP06))
            textfile.write('%d, %.4e, %.3f, %.3f, %.1f, %.4e\n' % (av, rel, totalVarP02, totalVarP06, tim, J_h))
        else:
            av, rel, J_h, integrand, tim = fixedMesh(i, op=op)
            print('Run %d: Mean element count: %6d Objective: %.4e OF error %.4e Timing %.1fs' % (i, av, J_h, rel, tim))
            textfile.write('%d, %.4e, %.1f, %.4e\n' % (av, rel, tim, J_h))
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
    textfile.close()
    integrandFile.close()
