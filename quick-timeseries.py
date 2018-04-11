import argparse

from utils.timeseries import plotTimeseries


parser = argparse.ArgumentParser()
parser.add_argument("mode",
                    help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave', 'model-verification'}.")
parser.add_argument("-r", help="Use rotational equations")
parser.add_argument("-l", help="Use linearised equations")
parser.add_argument("-d", help="Specify a date")
args = parser.parse_args()

for quantity in ('Integrand', 'P02', 'P06'):
    if args.mode == 'model-verification':
        fileExt = 'nonlinear='
        fileExt += 'False' if args.l else 'True'
        fileExt += '_rotational='
        fileExt += 'True' if args.r else 'False'
    else:
        raise NotImplementedError
    plotTimeseries(fileExt, date=args.d, mode=args.mode, quantity=quantity)
