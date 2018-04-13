import argparse

from utils.options import Options
from utils.timeseries import plotTimeseries, compareTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-approach",
                    help="Choose from {'fixedMesh', 'hessianBased', 'implicit', 'explicit', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Use rotational equations")
parser.add_argument("-l", help="Use linearised equations")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-c", help="Compare timeseries")
args = parser.parse_args()
op = Options(mode=args.mode)
approach = args.approach
if op.mode == 'model-verification':
    assert approach is None
if approach is None and op.mode != 'model-verification':
    approach = 'fixedMesh'
quantities = ['Integrand', 'P02', 'P06'] if op.mode in ('tohoku', 'model-verification') else ['Integrand']
for quantity in quantities:
    if op.mode == 'model-verification':
        fileExt = 'nonlinear='
        fileExt += 'False' if args.l else 'True'
        fileExt += '_rotational='
        fileExt += 'True' if args.r else 'False'
    else:
        fileExt = approach
    if args.c:
        for i in range(6):
            compareTimeseries(args.d, i, quantity=quantity, op=op)
    else:
        plotTimeseries(fileExt, date=args.d, quantity=quantity, op=op)
