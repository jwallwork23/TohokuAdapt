from firedrake import *

import argparse

from utils.options import *
from utils.setup import RossbyWaveSolution, problemDomain
from utils.timeseries import plotTimeseries, compareTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("t", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'kelvin-wave', 'model-verification'}.")
parser.add_argument("-a", help="Choose from {'fixedMesh', 'hessianBased', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Choose Coriolis parameter from {'off', 'f', 'beta', 'sin'}")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-c", help="Compare timeseries")
parser.add_argument("-g", help="Include actual gauge data")
parser.add_argument("-s", help="Generate rossby-wave analytic solution")
args = parser.parse_args()
approach = args.a
date = args.d
if args.t in ('tohoku', 'model-verification'):
    op = TohokuOptions(approach=approach)
elif args.t == 'shallow-water':
    op = GaussianOptions(approach=approach)
elif args.t == 'rossby-wave':
    op = RossbyWaveOptions(approach=approach)
elif args.t == 'kelvin-wave':
    op = KelvinWaveOptions(approach=approach)
elif args.t == 'advection-diffusion':
    op = AdvectionOptions(approach=approach)
if args.t == 'model-verification':
    assert approach is None
if approach is None and op.mode != 'model-verification':
    approach = 'fixedMesh'
if op.mode in ('tohoku', 'model-verification'):
    quantities = ['Integrand', 'P02', 'P06']
else:
    quantities = ['Integrand']
if bool(args.s):
    assert op.mode in ('rossby-wave', 'kelvin-wave')
    integrandFile = open('outdata/rossby-wave/analytic_Integrand.txt', 'w+')
    rw = RossbyWaveSolution(op.mixedSpace(problemDomain(level=7, op=op)[0]), order=1, op=op)
    integrand = rw.integrate()
    integrandFile.writelines(["%s," % val for val in integrand])
    integrandFile.write("\n")
    integrandFile.close()
    fileExt = 'analytic'
else:
    fileExt = approach
if not bool(args.s):
    if op.mode == 'model-verification':
        fileExt = '_rotational='
        fileExt += 'off' if args.r is None else args.r
    for quantity in quantities:
        if args.c:
            for i in range(6):
                compareTimeseries(date, i, quantity=quantity, op=op)
        else:
            plotTimeseries(fileExt, date=date, quantity=quantity, realData=bool(args.g), op=op)
