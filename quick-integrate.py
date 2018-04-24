import argparse

from utils.options import Options
from utils.timeseries import integrateTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-a",
                    help="Choose from {'fixedMesh', 'hessianBased', 'implicit', 'explicit', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Use rotational equations")
parser.add_argument("-l", help="Use linearised equations")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-s", help="Consider rossby-wave analytic solution")
args = parser.parse_args()
op = Options(mode=args.mode)
approach = args.a
if op.mode == 'model-verification':
    assert approach is None
if approach is None and op.mode != 'model-verification':
    approach = 'fixedMesh'
if op.mode == 'model-verification':
    fileExt = 'nonlinear='
    fileExt += 'False' if args.l else 'True'
    fileExt += '_rotational='
    fileExt += 'True' if args.r else 'False'
elif args.s is not None:
    assert op.mode == 'rossby-wave'
    fileExt = 'analytic'
else:
    fileExt = approach
print('Integral = ', integrateTimeseries(fileExt, date=args.d, op=op))
