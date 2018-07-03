import argparse

from utils.options import *
from utils.timeseries import integrateTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("t", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-a", help="Choose from {'fixedMesh', 'hessianBased', 'DWP', 'DWR'}.")
parser.add_argument("-r", help="Use rotational equations")
parser.add_argument("-d", help="Specify a date")
parser.add_argument("-s", help="Consider rossby-wave analytic solution")
parser.add_argument("-m", help="Consider 'mirror image' region of interest")
args = parser.parse_args()
mode = args.mode
approach = args.a
if mode in ('tohoku', 'model-verification'):
    op = TohokuOptions(approach=approach)
elif mode == 'rossby-wave':
    op = RossbyWaveOptions(approach=approach)
elif mode == 'kelvin-wave':
    op = KelvinWaveOptions(approach=approach)
elif mode == 'shallow-water':
    op = GaussianOptions(approach=approach)
elif mode == 'advection-diffusion':
    op = AdvectionOptions(approach=approach)
if op.mode == 'model-verification':
    assert approach is None
if approach is None and op.mode != 'model-verification':
    approach = 'fixedMesh'
if mode == 'model-verification':
    fileExt = 'rotational='
    fileExt += 'True' if args.r else 'False'
elif bool(args.s):
    assert op.mode == 'rossby-wave'
    fileExt = 'analytic'
else:
    fileExt = approach
print('Integral = ', integrateTimeseries(fileExt, date=args.d, op=op))
