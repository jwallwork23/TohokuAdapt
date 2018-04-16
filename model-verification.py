from thetis import *

from time import clock
import datetime
import numpy as np

from utils.callbacks import TohokuCallback, P02Callback, P06Callback
from utils.mesh import problemDomain
from utils.options import Options


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)


def solverSW(startRes, di, op=Options()):
    mesh, u0, eta0, b, BCs, f = problemDomain(level=startRes, op=op)

    # Get solver parameter values and construct solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True if op.nonlinear else False
    options.use_grad_depth_viscosity_term = False
    options.use_grad_div_viscosity_term = False
    options.coriolis_frequency = f
    options.simulation_export_time = 50. if op.plotpvd else 100.
    options.period_of_interest_start = op.Tstart
    options.simulation_end_time = op.Tend
    options.timestepper_type = op.timestepper
    options.timestep = op.dt
    if op.plotpvd:
        options.output_directory = di
    else:
        options.no_exports = True
    if op.printStats:
        options.timestepper_options.solver_parameters = {# TODO: Why is linear solver still taking 2 iterations?
                                                         # 'snes_monitor': True,
                                                         # 'snes_view': True,
                                                         'snes_converged_reason': True,
                                                         'ksp_converged_reason': True,
                                                        }
    # options.use_wetting_and_drying = op.wd        # TODO: Make this work
    # if op.wd:
    #     options.wetting_and_drying_alpha = alpha
    solver_obj.bnd_functions['shallow_water'] = BCs
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = TohokuCallback(solver_obj)        # Objective functional computation error
    solver_obj.add_callback(cb1, 'timestep')
    cb2 = P02Callback(solver_obj)           # Gauge timeseries error P02
    solver_obj.add_callback(cb2, 'timestep')
    cb3 = P06Callback(solver_obj)           # Gauge timeseries error P06
    solver_obj.add_callback(cb3, 'timestep')

    timer = clock()
    solver_obj.iterate()    # Run simulation
    timer = clock() - timer

    return cb1.quadrature(), cb1.__call__()[1], cb2.__call__()[1], cb2.totalVariation(), cb3.__call__()[1], \
           cb3.totalVariation(), timer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="Use rotational equations")
    parser.add_argument("-l", help="Use linearised equations")
    parser.add_argument("-w", help="Use wetting and drying")
    parser.add_argument("-o", help="Output data")
    parser.add_argument("-s", help="Print solver statistics")
    args = parser.parse_args()
    op = Options(family='dg-dg',
                 plotpvd=True if args.o else False,
                 printStats=True if args.s else False,
                 wd=True if args.w else False)
    op.nonlinear = False if args.l else True
    op.rotational = True if args.r else False
    tag = 'nonlinear=' + str(op.nonlinear) + '_' + 'rotational=' + str(op.rotational)
    if args.w:
        tag += '_w'
    filename = 'outdata/model-verification/' + tag + '_' + date
    errorfile = open(filename + '.txt', 'w+')
    gaugeFileP02 = open(filename + 'P02.txt', 'w+')
    gaugeFileP06 = open(filename + 'P06.txt', 'w+')
    integrandFile = open(filename + 'Integrand.txt', 'w+')
    di = 'plots/model-verification/' + tag + '/'

    resolutions = range(11)
    Jlist = np.zeros(len(resolutions))
    g2list = np.zeros(len(resolutions))
    g6list = np.zeros(len(resolutions))
    for k, i in zip(resolutions, range(len(resolutions))):
        print("\nStarting run %d... Nonlinear = %s, Rotational = %s\n" % (k, op.nonlinear, op.rotational))
        J_h, integrand, gP02, totalVarP02, gP06, totalVarP06, timing = solverSW(k, di, op=op)

        # Save to disk
        gaugeFileP02.writelines(["%s," % val for val in gP02])
        gaugeFileP02.write("\n")
        gaugeFileP06.writelines(["%s," % val for val in gP06])
        gaugeFileP06.write("\n")
        integrandFile.writelines(["%s," % val for val in integrand])
        integrandFile.write("\n")
        errorfile.write('%d, %.4e, %.4e, %.4e, %.1f\n' % (k, J_h, totalVarP02, totalVarP06, timing))
        print("\nRun %d... J_h: %.4e TV P02: %.3f, TV P06: %.3f, time: %.1f\n"
              % (k, J_h, totalVarP02, totalVarP06, timing))

        # Calculate orders of convergence
        Jlist[i] = J_h
        g2list[i] = totalVarP02
        g6list[i] = totalVarP06
        if i > 1:
            Jconv = (Jlist[i] - Jlist[i - 1]) / (Jlist[i - 1] - Jlist[i - 2])
            g2conv = (g2list[i] - g2list[i - 1]) / (g2list[i - 1] - g2list[i - 2])
            g6conv = (g6list[i] - g6list[i - 1]) / (g6list[i - 1] - g6list[i - 2])
            print("Orders of convergence... J: %.4f, P02: %.4f, P06: %.4f" % (Jconv, g2conv, g6conv))
    errorfile.close()
    gaugeFileP02.close()
    gaugeFileP06.close()
    integrandFile.close()
