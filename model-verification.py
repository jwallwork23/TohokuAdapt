from thetis import *
from firedrake.petsc import PETSc
from thetis.callback import DetectorsCallback

from time import clock
import h5py

from utils.callbacks import SWCallback
from utils.options import TohokuOptions
from utils.setup import problem_domain
from utils.timeseries import gauge_total_variation


def FixedMesh(mesh, u0, eta0, b, BCs={}, f=None, op=TohokuOptions()):

    # Get solver parameter values and construct solver
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.element_family = op.family
    options.use_nonlinear_equations = True
    # options.horizontal_viscosity = Constant(1e-3)
    options.use_grad_depth_viscosity_term = False
    options.use_grad_div_viscosity_term = True
    options.use_lax_friedrichs_velocity = False  # TODO: This is a temporary fix in adjoint case
    options.coriolis_frequency = f
    options.simulation_export_time = 50. if op.plot_pvd else 100.
    options.simulation_end_time = op.end_time
    options.timestepper_type = op.timestepper
    options.timestep = op.timestep
    if op.plot_pvd:
        options.output_directory = op.di
    else:
        options.no_exports = True
    solver_obj.bnd_functions['shallow_water'] = BCs
    solver_obj.assign_initial_conditions(elev=eta0, uv=u0)
    cb1 = SWCallback(solver_obj)        # Objective functional computation error
    cb1.op = op
    solver_obj.add_callback(cb1, 'timestep')
    gauges = ["P02", "P06"]
    cb2 = DetectorsCallback(solver_obj,
                            [op.gauge_coordinates(g) for g in gauges],
                            ['elev_2d' for g in gauges],
                            'timeseries',
                            gauges,
                            export_to_hdf5=True)
    solver_obj.add_callback(cb2, 'timestep')

    # Run simulation and extract quantities
    timer = clock()
    solver_obj.iterate()
    timer = clock() - timer

    quantities = {}
    quantities["J_h"] = cb1.get_val()
    hf = h5py.File(op.directory() + 'diagnostic_timeseries.hdf5', 'r')
    for g in gauges:
        quantities[g] = np.array(hf.get(g))
    hf.close()
    quantities["TV P02"] = gauge_total_variation(quantities["P02"], gauge="P02")
    quantities["TV P06"] = gauge_total_variation(quantities["P06"], gauge="P06")
    quantities["Element count"] = mesh.num_cells()
    quantities["Timer"] = timer

    return quantities


if __name__ == '__main__':
    import argparse
    import datetime
    import numpy as np


    now = datetime.datetime.now()
    date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", help="Output data")
    args = parser.parse_args()

    for c in ("off", "f", "beta", "sin"):
        op = TohokuOptions(plot_pvd=True if args.o else False,
                     coriolis=c)        # TODO: This won't currently work
        tag = 'rotational=' + c
        filename = 'outdata/model-verification/' + tag + '_' + date
        errorfile = open(filename + '.txt', 'w+')
        gaugeFileP02 = open(filename + 'P02.txt', 'w+')
        gaugeFileP06 = open(filename + 'P06.txt', 'w+')
        op.di = 'plots/model-verification/' + tag + '/'

        resolutions = range(11)
        Jlist = np.zeros(len(resolutions))
        g2list = np.zeros(len(resolutions))
        g6list = np.zeros(len(resolutions))
        for k, i in zip(resolutions, range(len(resolutions))):
            PETSc.Sys.Print("\nStarting run %d... Coriolis frequency: %s\n" % (k, c), comm=COMM_WORLD)
            mesh, u0, eta0, b, BCs, f, diffusivity = problem_domain(level=k, op=op)
            quantities = FixedMesh(mesh, u0, eta0, b, BCs, f, op=op)
            gaugeFileP02.writelines(["%s," % val for val in quantities["P02"]])
            gaugeFileP02.write("\n")
            gaugeFileP06.writelines(["%s," % val for val in quantities["P06"]])
            gaugeFileP06.write("\n")
            errorfile.write('%d, %.4e, %.4e, %.4e, %.1f\n'
                            % (k, quantities["J_h"], quantities["TV P02"], quantities["TV P06"], quantities["Timer"]))
            PETSc.Sys.Print("\nRun %d... J_h: %.4e TV P02: %.3f, TV P06: %.3f, time: %.1f\n"
                  % (k, quantities["J_h"], quantities["TV P02"], quantities["TV P06"], quantities["Timer"]), comm=COMM_WORLD)

            # Calculate orders of convergence
            Jlist[i] = quantities["J_h"]
            g2list[i] = quantities["TV P02"]
            g6list[i] = quantities["TV P06"]
            if i > 1:
                Jconv = (Jlist[i] - Jlist[i - 1]) / (Jlist[i - 1] - Jlist[i - 2])
                g2conv = (g2list[i] - g2list[i - 1]) / (g2list[i - 1] - g2list[i - 2])
                g6conv = (g6list[i] - g6list[i - 1]) / (g6list[i - 1] - g6list[i - 2])
                PETSc.Sys.Print("Orders of convergence... J: %.4f, P02: %.4f, P06: %.4f" % (Jconv, g2conv, g6conv),
                                comm=COMM_WORLD)
        errorfile.close()
        gaugeFileP02.close()
        gaugeFileP06.close()
