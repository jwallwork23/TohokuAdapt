from thetis_adjoint import *
from firedrake.petsc import PETSc

import argparse
import datetime
import numpy as np

from utils.options import TohokuOptions, GaussianOptions, RossbyWaveOptions, KelvinWaveOptions
from utils.setup import problem_domain
from utils.sw_solvers import tsunami


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

parser = argparse.ArgumentParser()
parser.add_argument("t", help="Choose test problem from {'shallow-water', 'rossby-wave'} (default 'tohoku')")
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
args = parser.parse_args()

approach = args.a
if approach is None:
    approach = 'FixedMesh'
else:
    assert approach in ('FixedMesh', 'HessianBased', 'DWP', 'DWR')
if args.t is None:
    mode = 'Tohoku'
else:
    mode = args.t
order_increase = False

# Establish filenames
filename = 'outdata/' + mode + '/' + approach
if args.ho:     # TODO: Difference quotient as alternative
    order_increase = True
    filename += '_ho'
if args.b:
    assert approach == 'HessianBased'
    filename += '_b'
filename += '_' + date
errorFile = open(filename + '.txt', 'w+')
files = {}
extensions = []
if mode == 'Tohoku':
    extensions.append('P02')
    extensions.append('P06')
for e in extensions:
    files[e] = open(filename + e + '.txt', 'w+')

# Set parameters
if mode == 'Tohoku':
    op = TohokuOptions(approach=approach)
elif mode == 'RossbyWave':
    op = RossbyWaveOptions(approach=approach)
elif mode == 'KelvinWave':
    op = KelvinWaveOptions(approach=approach)
elif mode == 'GaussianTest':
    op = GaussianOptions(approach=approach)
else:
    raise NotImplementedError
op.gradate = bool(args.g)
op.plot_pvd = bool(args.o)
op.plot_metric = bool(args.m)
op.adaptations = 1 if args.n is None else int(args.n)
op.order_increase = order_increase
op.adapt_on_bathymetry = bool(args.b)
op.adapt_field = args.f if args.f is not None else 's'
if bool(args.snes_view):
    op.solver_parameters['snes_view'] = True

# Get data and save to disk
if args.low is not None or args.high is not None:
    assert args.level is None
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
else:
    resolutions = [0 if args.level is None else int(args.level)]
Jlist = np.zeros(len(resolutions))
for i in resolutions:
    mesh, u0, eta0, b, BCs, f, diffusivity = problem_domain(i, op=op)
    quantities = tsunami(mesh, u0, eta0, b, BCs=BCs, f=f, diffusivity=diffusivity, regen=bool(args.regen), op=op)
    PETSc.Sys.Print("Mode: %s Approach: %s. Run: %d" % (mode, approach, i), comm=COMM_WORLD)
    rel = np.abs(op.J - quantities['J_h']) / np.abs(op.J)
    PETSc.Sys.Print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs OF error: %.4e"
          % (i, quantities['mean_elements'], quantities['J_h'], quantities['solver_timer'], rel), comm=COMM_WORLD)
    errorFile.write('%d, %.4e' % (quantities['mean_elements'], rel))
    for tag in ("peak", "dist", "spd", "TV P02", "TV P06"):
        if tag in quantities:
            errorFile.write(", %.4e" % quantities[tag])
    errorFile.write(", %.1f, %.4e\n" % (quantities['solver_timer'], quantities['J_h']))
    for tag in files:
        files[tag].writelines(["%s," % val for val in quantities[tag]])
        files[tag].write("\n")
    if approach in ("DWP", "DWR"):
        PETSc.Sys.Print("Time for final run: %.1fs" % quantities['adapt_solve_timer'], comm=COMM_WORLD)
for tag in files:
    files[tag].close()
errorFile.close()
