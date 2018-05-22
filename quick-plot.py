import argparse
import datetime

from utils.timeseries import errorVsElements


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-b", help="Specify bootstrapping")
parser.add_argument("-g", help="Include actual value (in Rossby wave case)")
parser.add_argument("-d", help="Specify a date (if different then type nothing at this stage)")
args = parser.parse_args()
if bool(args.g):
    assert args.mode == 'rossby-wave'
errorVsElements(args.mode, bootstrapping=bool(args.b), exact=bool(args.g), date=args.d)
