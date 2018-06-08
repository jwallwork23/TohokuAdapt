import argparse

from utils.options import TohokuOptions, RossbyWaveOptions
from utils.timeseries import errorVsElements


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-b", help="Specify bootstrapping")
parser.add_argument("-d", help="Specify a date (if different then type nothing at this stage)")
args = parser.parse_args()
if args.mode in ("tohoku", "model-verification"):
    op = TohokuOptions()
elif args.mode == "rossby-wave":
    op = RossbyWaveOptions()
errorVsElements(args.mode, bootstrapping=bool(args.b), date=args.d, op=op)
