from thetis import *

import scipy.interpolate as si
import matplotlib.pyplot as plt
import numpy as np
import datetime

from .options import Options


__all__ = ["readErrors", "extractSpline", "errorVsElements", "__main__", "plotTimeseries", "compareTimeseries"]


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')


def readErrors(date, approach, mode='tohoku', bootstrapping=False):
    """
    :arg date: date simulation was run.
    :arg approach: mesh adaptive approach.
    :param mode: problem considered.
    :param bootstrapping: toggle use of bootstrapping.
    :return: mean element count, (some aspect of) error and CPU time.
    """
    filename = 'outdata/'+mode+'/'+approach+date
    textfile = open(filename+'.txt', 'r')
    if mode == 'model-verification':
        bootstrapping = True
    nEls = []
    err = {}
    tim = []
    i = 0
    fixedMeshes = Options().meshSizes
    for line in textfile:
        if mode == 'model-verification':
            J_h, gP02, gP06, timing = line.split(',')[1:]
            nEls.append(fixedMeshes[i])
        else:
            nEls.append(int(av))
        if mode == 'tohoku':
            av, rel, gP02, gP06, timing, J_h = line.split(',')
        elif mode == 'shallow-water':
            av, rel, timing, J_h = line.split(',')
        elif mode == 'rossby-wave':
            av, rel, peak, dis, spd, timing, J_h = line.split(',')
        if bootstrapping:
            if mode == 'model-verification':
                err[i] = [float(J_h), float(gP02), float(gP06)]
            else:
                err[i] = [float(J_h)]
        else:
            if mode == 'tohoku':
                err[i] = [float(rel), float(gP02), float(gP06)]
            elif mode == 'shallow-water':
                err[i] = [float(rel)]
            elif mode == 'rossby-wave':
                err[i] = [float(rel), float(peak), float(dis), float(spd)]
        tim.append(float(timing))
        i += 1
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


def plotTimeseries(fileExt, date, quantity='Integrand', op=Options()):
    assert quantity in ('Integrand', 'P02', 'P06')
    filename = 'outdata/' + op.mode + '/' + fileExt + '_' + date + quantity + '.txt'
    f = open(filename, 'r')
    plt.gcf()
    i = 0
    for line in f:
        separated = line.split(',')
        dat = [float(d) for d in separated[:-1]]    # Ignore carriage return
        tim = np.linspace(0, op.Tend, len(dat))
        plt.plot(tim[::5], dat[::5], label=str(i))
        i += 1
    plt.xlabel('Time (s)')
    plt.ylabel(quantity+' value')
    plt.legend(loc=2)
    plt.savefig('outdata/' + op.mode + '/' + fileExt + '_' + quantity + date + '.pdf', bbox_inches='tight')
    plt.clf()


def compareTimeseries(date, run, quantity='Integrand', op=Options()):
    assert quantity in ('Integrand', 'P02', 'P06')
    approaches = ("fixedMesh", "hessianBased", "DWR")

    # Get dates (if necessary)
    dates = {}
    now = datetime.datetime.now()
    today = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
    for approach in approaches:
        if date is None:
            try:
                dates[approach] = input("Date to use for %s approach: " % approach)
            except:
                dates[approach] = today
        else:
            dates[approach] = date
    plt.gcf()
    for approach in approaches:
        filename = 'outdata/' + op.mode + '/' + approach + '_' + dates[approach] + quantity + '.txt'
        f = open(filename, 'r')
        for i in range(run):
            f.readline()
        separated = f.readline().split(',')
        dat = [float(d) for d in separated[:-1]]  # Ignore carriage return
        tim = np.linspace(0, op.Tend, len(dat))
        plt.plot(tim[::5], dat[::5], label=approach)
    plt.xlabel('Time (s)')
    plt.ylabel(quantity + ' value')
    plt.legend(loc=2)
    plt.savefig('outdata/' + op.mode + '/' + quantity + today + '_' +  str(run) + '.pdf', bbox_inches='tight')
    plt.clf()



def errorVsElements(mode='tohoku', bootstrapping=False, noTinyMeshes=True, date=None):
    if mode == 'model-verification':
        labels = ("Linear, non-rotational", "Linear, rotational", "Nonlinear, non-rotational", "Nonlinear, rotational")
        names = ("nonlinear=False_rotational=False_", "nonlinear=False_rotational=True_",
                 "nonlinear=True_rotational=False_", "nonlinear=True_rotational=True_")
    else:
        labels = ("Fixed mesh", "Hessian based", "Explicit", "Implicit", "DWF", "DWR", "Higher order DWR",
                  "Lower order DWR", "Refined DWR")
        names = ("fixedMesh", "hessianBased", "explicit", "implicit", "DWF", "DWR", "DWR_ho", "DWR_lo", "DWR_r")
    styles = {labels[0]: 's', labels[1]: '^', labels[2]: 'x', labels[3]: 'o'}
    if mode != 'model-verification':
        styles[labels[4]] = '*'
        styles[labels[5]] = 'h'
        styles[labels[6]] = 'v'
        styles[labels[7]] = '8'
        styles[labels[8]] = 's'
    err = {}
    nEls = {}
    tim = {}
    if mode == 'model-verification':
        bootstrapping = True

    if bootstrapping:
        errorlabels = [r'Objective value $J(\textbf{q})=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int\int_A'
                       +r'\eta(x,y,t)\,\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}t$']
        errornames = ['OF']
    else:
        errorlabels = [r'Relative error $\frac{|J(\textbf{q})-J(\textbf{q}_h)|}{|J(\textbf{q})|}$']
        errornames = ['rel']
    if mode in ('tohoku', 'model-verification'):
        errortypes = 3
        errorlabels.append('Relative total variation at gauge P02')
        errorlabels.append('Relative total variation at gauge P06')
        errornames.append('P02')
        errornames.append('P06')
    elif mode == 'shallow-water':
        errortypes = 1
    elif mode == 'rossby-wave':
        errortypes = 4
        errorlabels.append('Relative error in solition peak')
        errorlabels.append('Relative error in distance travelled')
        errorlabels.append('Relative error in phase speed')
        errornames.append('peak')
        errornames.append('dis')
        errornames.append('spd')

    # Get dates (if necessary)
    dates = []
    for n in range(len(names)):
        if date is None:
            try:
                dates.append(input("Date to use for %s approach: " % labels[n]))
            except:
                now = datetime.datetime.now()
                dates.append(str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000))
        else:
            dates.append(date)

    for m in range(errortypes):
        for n in range(len(names)):
            try:
                av, errors, timing = readErrors(dates[n], names[n], mode, bootstrapping=bootstrapping)
                rel = []
                for i in errors:
                    rel.append(errors[i][m])

                if noTinyMeshes:  # Remove results on meshes with very few elements
                    del av[0], av[1]
                    del rel[0], rel[1]
                    del timing[0], timing[1]

                err[labels[n]] = rel
                nEls[labels[n]] = av
                tim[labels[n]] = timing
            except:
                pass

        if err == {}:
            raise ValueError("No data available with these dates!")

        di = 'outdata/' + mode + '/'
        if bootstrapping:
            # Plot OF values
            for mesh in err:
                plt.semilogx(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
            plt.gcf()
            plt.legend(loc=1 if errornames[m] in ('P02', 'P06') else 4)
            plt.xlabel(r'Mean element count')
            plt.ylabel(errorlabels[m])
            plt.savefig(di + errornames[m] + 'VsElements' + date + '.pdf', bbox_inches='tight')
            plt.clf()
        else:
            # Plot errors
            for mesh in err:
                plt.loglog(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
            plt.gcf()
            plt.legend(loc=1)
            plt.xlabel(r'Mean element count')
            plt.ylabel(errorlabels[m])
            if mode == 'tohoku':
                plt.xlim([5000, 60000])
                plt.ylim([1e-4, 5e-1])
            plt.savefig(di + errornames[m] + 'VsElements' + date + '.pdf', bbox_inches='tight')
            plt.clf()

        # Plot timings
        if m == 0:
            for mesh in err:
                plt.loglog(nEls[mesh], tim[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
            plt.gcf()
            plt.legend(loc=2)
            plt.xlabel(r'Mean element count')
            plt.ylabel(r'CPU time (s)')
            if mode == 'tohoku':
                plt.xlim([0, 55000])
                plt.ylim([0, 5000])
            plt.savefig(di + 'timeVsElements' + date + '.pdf', bbox_inches='tight')
            plt.clf()
