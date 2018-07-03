# from thetis_adjoint import *
from thetis import *
from firedrake.petsc import PETSc

import argparse
import datetime
import numpy as np

from utils.options import AdvectionOptions
from utils.setup import problemDomain
from utils.ad_solvers import advect


now = datetime.datetime.now()
date = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)

parser = argparse.ArgumentParser()
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
parser.add_argument("-nAdapt", help="Number of mesh adaptation steps")
parser.add_argument("-gradate", help="Gradate metric")
parser.add_argument("-field", help="Field for Hessian based adaptation, from {'s', 'f', 'b'}.")
args = parser.parse_args()

approach = args.a
if approach is None:
    approach = 'fixedMesh'
else:
    assert approach in ('fixedMesh', 'hessianBased', 'DWP', 'DWR')
orderChange = 0
if args.ho:
    assert not args.r
    orderChange = 1
if args.r:
    assert not args.ho
if args.b is not None:
    assert approach == 'hessianBased'
coriolis = args.c if args.c is not None else 'f'

op = AdvectionOptions(approach=approach)
op.gradate = bool(args.gradate) if args.gradate is not None else False
op.plotpvd = True if args.o else False
op.nAdapt = 1 if args.nAdapt is None else int(args.nAdapt)
op.orderChange =  orderChange
op.bAdapt = bool(args.b) if args.b is not None else False
op.adaptField = args.field if args.field is not None else 's'

# Establish filenames
filename = 'outdata/advection-diffusion/' + approach
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

# Get data and save to disk
if args.low is not None or args.high is not None:
    assert args.level is None
    resolutions = range(0 if args.low is None else int(args.low), 6 if args.high is None else int(args.high))
else:
    resolutions = [0 if args.level is None else int(args.level)]
Jlist = np.zeros(len(resolutions))
for i in resolutions:
    mesh, u0, eta0, b, BCs, source, diffusivity = problemDomain(i, op=op)
    quantities = advect(mesh, u0, eta0, b, BCs=BCs, source=source, diffusivity=diffusivity, regen=bool(args.regen), op=op)
    PETSc.Sys.Print("Mode: %s Approach: %s. Run: %d" % ('advection-diffusion', approach, i), comm=COMM_WORLD)
    rel = np.abs(op.J - quantities['J_h']) / np.abs(op.J)
    PETSc.Sys.Print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs OF error: %.4e"
          % (i, quantities['meanElements'], quantities['J_h'], quantities['solverTimer'], rel), comm=COMM_WORLD)
    errorFile.write('%d, %.4e' % (quantities['meanElements'], rel))
    for tag in ("peak", "dist", "spd", "TV P02", "TV P06"):
        if tag in quantities:
            errorFile.write(", %.4e" % quantities[tag])
    errorFile.write(", %.1f, %.4e\n" % (quantities['solverTimer'], quantities['J_h']))
    if approach in ("DWP", "DWR"):
        PETSc.Sys.Print("Time for final run: %.1fs" % quantities['adaptSolveTimer'], comm=COMM_WORLD)
errorFile.close()
