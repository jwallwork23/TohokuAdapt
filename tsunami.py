from thetis_adjoint import *
from firedrake.petsc import PETSc

import argparse
import datetime
import numpy as np

from utils.options import Options
from utils.solvers import fixedMesh, hessianBased, DWP, DWR


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

parser = argparse.ArgumentParser()
parser.add_argument("t", help="Choose test problem from {'shallow-water', 'rossby-wave'} (default 'tohoku')")
parser.add_argument("-a", help="Choose adaptive approach from {'hessianBased', 'DWP', 'DWR'} (default 'fixedMesh')")
parser.add_argument("-low", help="Lower bound for mesh resolution range")
parser.add_argument("-high", help="Upper bound for mesh resolution range")
parser.add_argument("-level", help="Single mesh resolution")
parser.add_argument("-ho", help="Compute errors and residuals in a higher order space")
parser.add_argument("-r", help="Compute errors and residuals in a refined space")
parser.add_argument("-b", help="Intersect metrics with bathymetry")
parser.add_argument("-c", help="Type of Coriolis coefficient to use, from {'off', 'f', 'beta', 'sin'}.")
parser.add_argument("-o", help="Output data")
parser.add_argument("-regen", help="Regenerate error estimates from saved data")
parser.add_argument("-mirror", help="Use a 'mirrored' region of interest")
parser.add_argument("-nAdapt", help="Number of mesh adaptation steps")
args = parser.parse_args()

solvers = {'fixedMesh': fixedMesh, 'hessianBased': hessianBased, 'DWP': DWP, 'DWR': DWR}
approach = args.a
if approach is None:
    approach = 'fixedMesh'
else:
    assert approach in solvers.keys()
solver = solvers[approach]
if args.t is None:
    mode = 'tohoku'
else:
    mode = args.t
orderChange = 0
if args.ho:
    assert not args.r
    orderChange = 1
if args.r:
    assert not args.ho
if args.b is not None:
    assert approach == 'hessianBased'
if bool(args.mirror):
    assert mode in ('shallow-water', 'rossby-wave')
coriolis = args.c if args.c is not None else 'f'

# Choose mode and set parameter values
op = Options(mode=mode,
             approach=approach,
             gradate=True if approach in ('DWP', 'DWR') and mode == 'tohoku' else False,
             plotpvd=True if args.o else False,
             coriolis=coriolis,
             nAdapt=1 if args.nAdapt is None else int(args.nAdapt),
             orderChange=orderChange,
             refinedSpace=True if args.r else False,
             bAdapt=bool(args.b) if args.b is not None else False)

# Establish filenames
filename = 'outdata/' + mode + '/' + approach
if args.ho:
    op.orderChange = 1
    filename += '_ho'
elif args.r:
    op.refinedSpace = True
    filename += '_r'
if args.b:
    filename += '_b'
filename += '_' + date
errorFile = open(filename + '.txt', 'w+')
files = {}
extensions = ['Integrand']
if op.mode == 'tohoku':
    extensions.append('P02')
    extensions.append('P06')
else:
    extensions.append('Integrand-mirrored')
for e in extensions:
    files[e] = open(filename + e + '.txt', 'w+')

# Get data and save to disk
if args.low is not None or args.high is not None:
    assert args.level is None
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
else:
    resolutions = [0 if args.level is None else int(args.level)]
Jlist = np.zeros(len(resolutions))
for i in resolutions:
    quantities = solver(i, regen=bool(args.regen), mirror=bool(args.mirror), op=op)
    PETSc.Sys.Print("Mode: %s Approach: %s. Run: %d" % (mode, approach, i), comm=COMM_WORLD)
    rel = np.abs(op.J - quantities['J_h']) / np.abs(op.J)
    if op.mode == "rossby-wave":
        quantities["Mirrored OF error"] = np.abs(op.J_mirror - quantities['J_h mirrored']) / np.abs(op.J_mirror)
    PETSc.Sys.Print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs OF error: %.4e"
          % (i, quantities['meanElements'], quantities['J_h'], quantities['solverTimer'], rel), comm=COMM_WORLD)
    errorFile.write('%d, %.4e' % (quantities['meanElements'], rel))
    for tag in ("peak", "dist", "spd", "TV P02", "TV P06", "J_h mirrored", "Mirrored OF error"):
        if tag in quantities:
            errorFile.write(", %.4e" % quantities[tag])
    errorFile.write(", %.1f, %.4e\n" % (quantities['solverTimer'], quantities['J_h']))
    for tag in files:
        files[tag].writelines(["%s," % val for val in quantities[tag]])
        files[tag].write("\n")
    if approach in ("DWP", "DWR"):
        PETSc.Sys.Print("Time for final run: %.1fs" % quantities['adaptSolveTimer'], comm=COMM_WORLD)
for tag in files:
    files[tag].close()
errorFile.close()
