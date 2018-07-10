import argparse

from utils.options import *
from utils.timeseries import error_vs_elements


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-b", help="Specify bootstrapping")
parser.add_argument("-d", help="Specify a date (if different then type nothing at this stage)")
args = parser.parse_args()
if args.mode in ("tohoku", "model-verification"):
    op = TohokuOptions()
elif args.mode == "rossby-wave":
    op = RossbyWaveOptions()
elif args.mode == "advection-diffusion":
    op = AdvectionOptions()
error_vs_elements(args.mode, bootstrapping=bool(args.b), date=args.d, op=op)
