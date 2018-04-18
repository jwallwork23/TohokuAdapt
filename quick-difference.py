import argparse

from utils.options import Options
from utils.timeseries import timeseriesDifference


parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-approach1",
                    help="Choose from {'fixedMesh', 'hessianBased', 'implicit', 'explicit', 'DWP', 'DWR'}.")
parser.add_argument("-approach2",
                    help="Choose from {'fixedMesh', 'hessianBased', 'implicit', 'explicit', 'DWP', 'DWR'}.")
parser.add_argument("-r1", help="Use rotational equations for first timeseries")
parser.add_argument("-l1", help="Use linearised equations for first timeseries")
parser.add_argument("-r2", help="Use rotational equations for second timeseries")
parser.add_argument("-l2", help="Use linearised equations for second timeseries")
parser.add_argument("-d1", help="Specify first date")
parser.add_argument("-d2", help="Specify second date")
args = parser.parse_args()
op = Options(mode=args.mode)
approach1 = args.approach1
approach2 = args.approach2
if op.mode == 'model-verification':
    assert approach1 is None and approach2 is None
if approach1 is None and op.mode != 'model-verification':
    approach1 = 'fixedMesh'
if approach2 is None and op.mode != 'model-verification':
    approach2 = 'fixedMesh'
quantities = ['Integrand', 'P02', 'P06'] if op.mode in ('tohoku', 'model-verification') else ['Integrand']
for quantity in quantities:
    if op.mode == 'model-verification':
        fileExt1 = 'nonlinear='
        fileExt1 += 'False' if args.l1 else 'True'
        fileExt1 += '_rotational='
        fileExt1 += 'True' if args.r1 else 'False'
        fileExt2 = 'nonlinear='
        fileExt2 += 'False' if args.l2 else 'True'
        fileExt2 += '_rotational='
        fileExt2 += 'True' if args.r2 else 'False'
    else:
        fileExt1 = approach1
        fileExt2 = approach2
    print(quantity, " error = ", timeseriesDifference(fileExt1, args.d1, fileExt2, args.d2, quantity=quantity, op=op))
