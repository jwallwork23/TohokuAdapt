from firedrake import *

import argparse

from utils.options import *
from utils.timeseries import plot_timeseries, compare_timeseries


parser = argparse.ArgumentParser()
parser.add_argument("-a", help="Choose from {'FixedMesh', 'HessianBased', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Choose Coriolis parameter from {'off', 'f', 'beta', 'sin'}")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-g", help="Include actual gauge data")
args = parser.parse_args()
approach = args.a
date = args.d
op = TohokuOptions(approach=approach)
if approach is None:
    approach = 'FixedMesh'
quantities = ['P02', 'P06']
fileExt = approach
for quantity in quantities:
    if args.c:
        for i in range(6):
            compare_timeseries(date, i, quantity=quantity, op=op)
    else:
        plot_timeseries(fileExt, date=date, quantity=quantity, realData=bool(args.g), op=op)
