from firedrake import *

from time import clock

from utils.adaptivity import *
from utils.interpolation import interp
from utils.misc import indicator, indexString
from utils.options import Options

from solvers.apriori import fixedMeshAD, hessianBasedAD
from utils.options import Options


def DWP():
    raise NotImplementedError


def DWR():
    raise NotImplementedError


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("-a", help="Choose adaptive approach from {'hessianBased', 'DWP', 'DWR'} (default 'fixedMesh')")
    parser.add_argument("-low", help="Lower bound for index range")
    parser.add_argument("-high", help="Upper bound for index range")
    parser.add_argument("-level", help="Single mesh resolution")
    parser.add_argument("-nAdapt")
    parser.add_argument("-o", help="Output data")
    args = parser.parse_args()

    op = Options(mode="advection-diffusion",
                 approach='fixedMesh' if args.a is None else args.a,
                 plotpvd=True if bool(args.o) else False,
                 nAdapt= int(args.nAdapt) if args.nAdapt is not None else 1)

    # Get data and save to disk
    if args.low is not None or args.high is not None:
        assert args.level is None
        resolutions = range(0 if args.low is None else int(args.low), 8 if args.high is None else int(args.high))
    else:
        resolutions = [0 if args.level is None else int(args.level)]
    solvers = {'fixedMesh': fixedMeshAD, 'hessianBased': hessianBasedAD}

    errorFile = open('outdata/advection-diffusion/' + op.approach + '.txt', 'w+')
    for i in resolutions:
        q = solvers[op.approach](pow(2, i), op=op)
        print("Run %d: Mean element count: %6d Objective: %.4e Timing %.1fs"
              % (i, q['meanElements'], q['J_h'], q['solverTimer']))
        errorFile.write('%d, %.4e\n' % (q['meanElements'], q['J_h']))
    errorFile.close()
