import argparse

from utils.options import *
from utils.timeseries import error_vs_elements


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Choose problem from {'Tohoku', 'GaussianTest', 'RossbyWave', 'KelvinWave', 'AdvectionDiffusion'}.")
parser.add_argument("-b", help="Specify bootstrapping")
parser.add_argument("-d", help="Specify a date (if different then type nothing at this stage)")
args = parser.parse_args()
if args.mode == "Tohoku":
    op = TohokuOptions()
elif args.mode == "GaussianTest":
    op = GaussianOptions()
elif args.mode == "RossbyWave":
    op = RossbyWaveOptions()
elif args.mode == "KelvinWave":
    op = KelvinWaveOptions()
elif args.mode == "AdvectionDiffusion":
    op = AdvectionOptions()
error_vs_elements(args.mode, bootstrapping=bool(args.b), date=args.d, op=op)
