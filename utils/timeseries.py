from thetis import *

import scipy.interpolate as si
import matplotlib.pyplot as plt


__all__ = ["readErrors", "extractSpline", "errorVsElements", "__main__"]


def readErrors(date, approach, mode='tohoku', bootstrapping=False):
    """
    :arg date: date simulation was run.
    :arg approach: mesh adaptive approach.
    :param mode: problem considered.
    :param bootstrapping: was bootstrapping used?
    :return: mean element count, relative error and CPU time.
    """
    filename = 'outdata/'+mode+'/'+approach+date
    textfile = open(filename+'.txt', 'r')
    nEls = []
    err = []
    tim = []
    for line in textfile:
        av, rel, timing, J_h = line.split(',')    # TODO: rossby-wave case has different format
        nEls.append(int(av))                      # TODO: so is model-verification
        if bootstrapping:
            err.append(float(J_h))
        else:
            err.append(float(rel))
        tim.append(float(timing))
    textfile.close()
    return nEls, err, tim


def extractSpline(gauge):
    measuredFile = open('resources/gauges/'+gauge+'data_25mins.txt', 'r')
    x = []
    y = []
    for line in measuredFile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    spline = si.interp1d(x, y, kind=1)
    measuredFile.close()
    return spline


def errorVsElements(mode='tohoku', modelVerif=False, bootstrapping=False, noTinyMeshes=True, date=None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    if modelVerif:
        labels = ("Linear, non-rotational", "Linear, rotational", "Nonlinear, non-rotational", "Nonlinear, rotational")
        names = ("nonlinear=False_rotational=False_", "nonlinear=False_rotational=True_",
                 "nonlinear=True_rotational=False_", "nonlinear=True_rotational=True_")
    else:
        labels = ("Fixed mesh", "Hessian based", "Explicit", "Implicit", "DWF", "DWR", "Higher order DWR",
                  "Lower order DWR", "Refined DWR")
        names = ("fixedMesh", "hessianBased", "explicit", "implicit", "DWF", "DWR", "DWR_ho", "DWR_lo", "DWR_r")
    styles = {labels[0]: 's', labels[1]: '^', labels[2]: 'x', labels[3]: 'o'}
    if not modelVerif:
        styles[labels[4]] = '*'
        styles[labels[5]] = 'h'
        styles[labels[6]] = 'v'
        styles[labels[7]] = '8'
        styles[labels[8]] = 's'
    err = {}
    nEls = {}
    tim = {}
    for i in range(len(names)):
        try:
            if date is not None:
                av, rel, timing = readErrors(date, names[i], mode, bootstrapping=bootstrapping)
            else:
                av, rel, timing = readErrors(input("Date to use for %s approach: " % labels[i]), names[i], mode,
                                             bootstrapping=bootstrapping)
            if noTinyMeshes:        # Remove results on meshes with very few elements
                del av[0], av[1]
                del rel[0], rel[1]
                del timing[0], timing[1]
            err[labels[i]] = rel
            nEls[labels[i]] = av
            tim[labels[i]] = timing
        except:
            pass

    di = 'outdata/'
    if modelVerif:
        di += 'model-verification/'
    else:
        di += mode + '/'
    if bootstrapping:
        # Plot OF values
        for mesh in err:
            plt.semilogx(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        plt.gcf()
        plt.legend(loc=4)
        plt.xlabel(r'Mean element count')
        plt.ylabel(r'Objective value $J(\textbf{q})=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int\int_A'
                   +r'\eta(x,y,t)\,\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}t$')
        plt.savefig(di + 'objectiveVsElements.pdf', bbox_inches='tight')
        plt.clf()
    else:
        # Plot errors
        for mesh in err:
            plt.loglog(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        plt.gcf()
        plt.legend(loc=1)
        plt.xlabel(r'Mean element count')
        plt.ylabel(r'Relative error $\frac{|J(\textbf{q})-J(\textbf{q}_h)|}{|J(\textbf{q})|}$')
        if mode == 'tohoku':
            plt.xlim([5000, 60000])
            plt.ylim([1e-4, 5e-1])
        plt.savefig(di + 'errorVsElements.pdf', bbox_inches='tight')
        plt.clf()

    # Plot timings
    for mesh in err:
        plt.loglog(nEls[mesh], tim[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
    plt.gcf()
    plt.legend(loc=2)
    plt.xlabel(r'Mean element count')
    plt.ylabel(r'CPU time (s)')
    if mode == 'tohoku':
        plt.xlim([0, 55000])
        plt.ylim([0, 5000])
    plt.savefig(di + 'timeVsElements.pdf', bbox_inches='tight')


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Choose problem from {'tohoku', 'shallow-water', 'rossby-wave'}.")
    parser.add_argument("-mv", help="Specify use of model verification")
    parser.add_argument("-b", help="Specify bootstrapping")
    parser.add_argument("-d", help="Specify a date")
    args = parser.parse_args()
    errorVsElements(args.mode, modelVerif=args.mv, bootstrapping=args.b, date=args.d)
