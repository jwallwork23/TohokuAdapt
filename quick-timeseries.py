from firedrake import *

import argparse

from utils.options import Options
from utils.setup import integrateRW, problemDomain
from utils.timeseries import plotTimeseries, compareTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-a", help="Choose from {'fixedMesh', 'hessianBased', 'implicit', 'explicit', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Choose Coriolis parameter from {'off', 'f', 'beta', 'sin'}")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-c", help="Compare timeseries")
parser.add_argument("-g", help="Include actual gauge data")
parser.add_argument("-s", help="Consider rossby-wave analytic solution")
args = parser.parse_args()
approach = args.a
date = args.d
op = Options(mode=args.mode, approach=approach)
if op.mode == 'model-verification':
    assert approach is None
if approach is None and op.mode != 'model-verification':
    approach = 'fixedMesh'
quantities = ['Integrand', 'P02', 'P06'] if op.mode in ('tohoku', 'model-verification') else ['Integrand']
if args.s is not None:
    assert op.mode == 'rossby-wave'
    integrandFile = open('outdata/' + op.mode + '/analytic_Integrand.txt', 'w+')
    integrand = integrateRW(op.mixedSpace(problemDomain(level=5, op=op)[0]), op=op)
    integrandFile.writelines(["%s," % val for val in integrand])
    integrandFile.write("\n")
    integrandFile.close()
if op.mode == 'model-verification':
    fileExt = '_rotational='
    fileExt += 'off' if args.r is None else args.r
elif args.s is not None:
    assert op.mode == 'rossby-wave'
    fileExt = 'analytic'
else:
    fileExt = approach
for quantity in quantities:
    if args.c:
        for i in range(6):
            compareTimeseries(date, i, quantity=quantity, op=op)
    else:
        plotTimeseries(fileExt, date=date, quantity=quantity, realData=bool(args.g), op=op)
