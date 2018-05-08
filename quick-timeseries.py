from firedrake import *

import argparse

from utils.options import Options
from utils.setup import RossbyWaveSolution, problemDomain
from utils.timeseries import plotTimeseries, compareTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("t", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-a", help="Choose from {'fixedMesh', 'hessianBased', 'implicit', 'explicit', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Choose Coriolis parameter from {'off', 'f', 'beta', 'sin'}")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-c", help="Compare timeseries")
parser.add_argument("-g", help="Include actual gauge data")
parser.add_argument("-s", help="Generate rossby-wave analytic solution")
parser.add_argument("-m", help="Consider 'mirror image' region of interest")
args = parser.parse_args()
approach = args.a
date = args.d
op = Options(mode=args.t, approach=approach)
if op.mode == 'model-verification':
    assert approach is None
if approach is None and op.mode != 'model-verification':
    approach = 'fixedMesh'
quantities = ['Integrand', 'P02', 'P06'] if op.mode in ('tohoku', 'model-verification') else ['Integrand']
if bool(args.s):
    assert op.mode == 'rossby-wave'
    integrandFile = open('outdata/' + op.mode + '/analytic_Integrand.txt', 'w+')
    integrand = RossbyWaveSolution(op.mixedSpace(problemDomain(level=6, op=op)[0]), order=1, op=op).integrate(bool(args.m))
    integrandFile.writelines(["%s," % val for val in integrand])
    integrandFile.write("\n")
    integrandFile.close()
    fileExt = 'analytic'
else:
    fileExt = approach
if op.mode == 'model-verification':
    fileExt = '_rotational='
    fileExt += 'off' if args.r is None else args.r
if bool(args.m):
    assert bool(args.s) and op.mode in ('shallow-water', 'rossby-wave')
    fileExt += '_mirror'
for quantity in quantities:
    if args.c:
        for i in range(6):
            compareTimeseries(date, i, quantity=quantity, op=op)
    else:
        plotTimeseries(fileExt, date=date, quantity=quantity, realData=bool(args.g), op=op)
