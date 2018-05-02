import argparse
import datetime

from utils.timeseries import errorVsElements


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-b", help="Specify bootstrapping")
parser.add_argument("-d", help="Specify a date (if different then type nothing at this stage)")
args = parser.parse_args()
errorVsElements(args.mode, bootstrapping=args.b, date=args.d)
