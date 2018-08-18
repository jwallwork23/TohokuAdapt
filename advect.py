from thetis_adjoint import *
from firedrake.petsc import PETSc

import argparse
import datetime
import numpy as np

from utils.options import AdvectionOptions
from utils.setup import problem_domain
from utils.ad_solvers import advect


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

parser = argparse.ArgumentParser()
parser.add_argument("-a", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh')")
parser.add_argument("-f", help="Finite element family, from {'dg', 'cg'}")
parser.add_argument("-g", help="Gradate metric")
parser.add_argument("-m", help="Output metric data")
parser.add_argument("-n", help="Number of mesh adaptation steps")
parser.add_argument("-o", help="Output data")
parser.add_argument("-low", help="Lower bound for mesh resolution range")
parser.add_argument("-high", help="Upper bound for mesh resolution range")
parser.add_argument("-level", help="Single mesh resolution")
parser.add_argument("-regen", help="Regenerate error estimates from saved data")
parser.add_argument("-snes_view", help="Use PETSc snes view.")
parser.add_argument("-nodebug", help="Hide error messages.")
args = parser.parse_args()

approach = args.a
if args.a is None:
    approaches = ['FixedMesh']
elif args.a == 'all':
    approaches = ['FixedMesh', 'HessianBased', 'DWP', 'DWR']
else:
    assert args.a in ['FixedMesh', 'HessianBased', 'DWP', 'DWR']
    approaches = [args.a]
failures = []
nodebug = False if args.nodebug is None else bool(args.nodebug)

for approach in approaches:
    errorFile = open('outdata/AdvectionDiffusion/' + approach + '_' + date + '.txt', 'w+')

    # Set parameters
    op = AdvectionOptions(approach=approach)
    op.gradate = bool(args.g)
    op.plot_pvd = bool(args.o)
    op.plot_metric = bool(args.m)
    op.tracer_family = args.f if args.f is not None else 'cg'
    op.num_adapt = 1 if args.n is None else int(args.n)
    if bool(args.snes_view):
        op.solver_parameters['snes_view'] = True
    op.order_increase = True    # TODO: difference quotient option

    # TODO: continue testing
    if op.approach in ("DWP", "DWR"):
        op.rescaling = 0.1

    # Get data and save to disk
    if args.low is not None or args.high is not None:
        assert args.level is None
        resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
    else:
        resolutions = [0 if args.level is None else int(args.level)]
    Jlist = np.zeros(len(resolutions))
    for i in resolutions:

        def run_model():
            mesh, u0, eta0, b, BCs, source, diffusivity = problem_domain(i, op=op)
            quantities = advect(mesh, u0, eta0, b, BCs=BCs, source=source, diffusivity=diffusivity,
                                regen=bool(args.regen), op=op)
            PETSc.Sys.Print("Mode: %s Approach: %s. Run: %d" % ('advection-diffusion', approach, i), comm=COMM_WORLD)
            rel = np.abs(op.J - quantities['J_h']) / np.abs(op.J)
            PETSc.Sys.Print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs OF error: %.4e"
                            % (i, quantities['mean_elements'], quantities['J_h'], quantities['solver_timer'], rel),
                            comm=COMM_WORLD)
            errorFile.write('%d, %.4e' % (quantities['mean_elements'], rel))
            for tag in ("peak", "dist", "spd", "TV P02", "TV P06"):
                if tag in quantities:
                    errorFile.write(", %.4e" % quantities[tag])
            errorFile.write(", %.1f, %.4e\n" % (quantities['solver_timer'], quantities['J_h']))

            if op.plot_cross_section:
                import matplotlib.pyplot as plt

                i_end = op.final_export()
                for progress in (0.5, 1):
                    tag = "h_snapshot_" + str(int(i_end * progress))  # TODO: This is for non-diffusive case
                    if tag in quantities:  # TODO: Consider steady state for diffusive case
                        plt.clf()
                        s = quantities[tag]
                        sl = op.h_slice
                        x = np.linspace(sl[0][0], sl[-1][0], len(sl))
                        plt.plot(x, s)
                        plt.title("Tracer concentration at time %.1fs" % (op.end_time * progress))
                        plt.xlabel("Abcissa (m)")
                        plt.ylabel("Tracer concentraton (g/L)")
                        plt.savefig('outdata/AdvectionDiffusion/' + tag + '.pdf')
                for progress in (0.5, 1):
                    tag = "v_snapshot_" + str(int(i_end * progress))
                    if tag in quantities:
                        plt.clf()
                        s = quantities[tag]
                        sl = op.v_slice
                        x = np.linspace(sl[0][0], sl[0][-1], len(sl))
                        plt.plot(x, s)
                        plt.title("Tracer concentration at time %.1fs" % (op.end_time * progress))
                        plt.xlabel("Ordinate (m)")
                        plt.ylabel("Tracer concentration (g/L)")
                        plt.savefig('outdata/AdvectionDiffusion/' + tag + '.pdf')
            if approach in ("DWP", "DWR"):
                PETSc.Sys.Print("Time for final run: %.1fs" % quantities['adapt_solve_timer'], comm=COMM_WORLD)

        if nodebug:
            try:
                run_model()
            except:
                PETSc.Sys.Print("WARNING: %s run %d failed!" % (op.approach, i), comm=COMM_WORLD)
                failures.append("{a:s} run {r:d}".format(a=approach, r=i))
        else:
            run_model()
    errorFile.close()

if failures != []:
    print("Failure summary:")
    for f in failures:
        print(f)
