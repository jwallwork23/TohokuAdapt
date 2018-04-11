import argparse

from utils.timeseries import plotTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("quantity", help="Choose quantity of interest from {'Integrand', 'P02', 'P06'}.")
parser.add_argument("-r", help="Use rotational equations")
parser.add_argument("-l", help="Use linearised equations")
parser.add_argument("-d", help="Specify a date")
args = parser.parse_args()
if args.mode == 'model-verification':
    filename = 'outdata/' + args.mode + '/nonlinear='
    filename += 'False' if args.l else 'True'
    filename += '_rotational='
    filename += 'True' if args.r else 'False'
    filename += '_' + args.d + args.quantity + '.txt'
plotTimeseries(filename, date=args.d, mode=args.mode, quantity=args.quantity)
