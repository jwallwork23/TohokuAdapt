from thetis_adjoint import *
from firedrake.petsc import PETSc

import argparse
import datetime
import numpy as np

from utils.options import TohokuOptions
from utils.setup import problem_domain
from utils.sw_solvers import tsunami


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

parser = argparse.ArgumentParser()
parser.add_argument("-a", help="Choose adaptive approach from {'HessianBased', 'DWP', 'DWR'} (default 'FixedMesh')")
parser.add_argument("-b", help="Intersect metrics with bathymetry")
parser.add_argument("-f", help="Field for Hessian based adaptation, from {'s', 'f', 'b'}.")
parser.add_argument("-g", help="Gradate metric")
parser.add_argument("-ho", help="Compute errors and residuals in a higher order space")
parser.add_argument("-m", help="Output metric data")
parser.add_argument("-n", help="Number of mesh adaptation steps")
parser.add_argument("-o", help="Output data")
parser.add_argument("-low", help="Lower bound for mesh resolution range")
parser.add_argument("-high", help="Upper bound for mesh resolution range")
parser.add_argument("-level", help="Single mesh resolution")
parser.add_argument("-regen", help="Regenerate error estimates from saved data")
parser.add_argument("-snes_view", help="Use PETSc snes sview.")
parser.add_argument("-nodebug", help="Hide error messages.")
args = parser.parse_args()

order_increase = True  # TODO: difference quotient option
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

    # Establish filenames
    filename = 'outdata/Tohoku/' + approach
    if approach == 'HessianBased':
        field_for_adaptation = args.f if args.f is not None else 's'
        filename += '_' + field_for_adaptation
    filename += '_' + date
    files = {}
    extensions = ['P02', 'P06']
    for e in extensions:
        files[e] = open(filename + e + '.txt', 'w+')

    # Set parameters
    op = TohokuOptions(approach=approach)
    op.gradate = bool(args.g)
    op.plot_pvd = bool(args.o)
    op.plot_metric = bool(args.m)
    op.num_adapt = 1 if args.n is None else int(args.n)
    op.order_increase = order_increase
    op.adapt_on_bathymetry = bool(args.b)
    if approach == 'HessianBased':
        op.adapt_field = field_for_adaptation
    if bool(args.snes_view):
        op.solver_parameters['snes_view'] = True

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
            errorFile = open(filename + '.txt', 'a+')   # Append mode ensures against crashes meaning losing all data
            mesh, u0, eta0, b, BCs, f, diffusivity = problem_domain(i, op=op)
            quantities = tsunami(mesh, u0, eta0, b, BCs=BCs, f=f, diffusivity=diffusivity, regen=bool(args.regen), op=op)
            PETSc.Sys.Print("Mode: Tohoku Approach: %s. Run: %d" % (approach, i), comm=COMM_WORLD)
            PETSc.Sys.Print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs"
                  % (i, quantities['mean_elements'], quantities['J_h'], quantities['solver_timer']), comm=COMM_WORLD)
            errorFile.write('%d,%.1f,%.4e' % (quantities['mean_elements'], quantities['solver_timer'],quantities['J_h']))
            for tag in ("TV P02", "TV P06"):
                if tag in quantities:
                    errorFile.write(", %.4e" % quantities[tag])
            errorFile.write("\n")
            for tag in files:
                files[tag].writelines(["%s," % val for val in quantities[tag]])
                files[tag].write("\n")
            if approach in ("DWP", "DWR"):
                PETSc.Sys.Print("Time for final run: %.1fs" % quantities['adapt_solve_timer'], comm=COMM_WORLD)
            errorFile.close()

        if nodebug:
            try:
                run_model()
            except:
                PETSc.Sys.Print("WARNING: %s run %d failed!" % (op.approach, i), comm=COMM_WORLD)
                failures.append("{a:s} run {r:d}".format(a=approach, r=i))
        else:
            run_model()
    for tag in files:
        files[tag].close()

if failures != []:
    print("Failure summary:")
    for f in failures:
        print(f)
