from thetis import *

import scipy.interpolate as si
import matplotlib.pyplot as plt
import numpy as np
import datetime

from .options import TohokuOptions, AdvectionOptions


__all__ = ["error_vs_elements", "plot_timeseries", "compare_timeseries", "timeseries_difference", "integrate_timeseries"]


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')


def read_errors(date, approach, mode='Tohoku'):
    """
    :arg date: date simulation was run.
    :arg approach: mesh adaptive approach.
    :param mode: problem considered.
    :return: mean element count, (some aspects of) error and CPU time.
    """
    filename = 'outdata/'+mode+'/'+approach+'_'+date
    textfile = open(filename+'.txt', 'r')
    nEls = []
    err = {}
    tim = []
    i = 0
    for line in textfile:
        if mode == 'Tohoku':
            av, timing, J_h, gP02, gP06 = line.split(',')
        elif mode == 'AdvectionDiffusion':
            av, timing, J_h = line.split(',')
        nEls.append(int(av))
        if mode == 'Tohoku':
            err[i] = [float(gP02), float(gP06)]
        tim.append(float(timing))
        i += 1
    textfile.close()
    return nEls, err, tim


def extract_data(gauge):
    if gauge in ("P02", "P06"):
        measuredFile = open('resources/gauges/' + gauge + 'data_25mins.txt', 'r')
        x = []
        y = []
        for line in measuredFile:
            xy = line.split()
            x.append(float(xy[0]))
            y.append(float(xy[1]))

        return x, y
    else:
        raise NotImplementedError


def plot_timeseries(fileExt, date, realData=False, op=TohokuOptions()):
    assert quantity in ('P02', 'P06')
    filename = 'outdata/' + op.mode + '/' + fileExt + '_' + date + quantity + '.txt'
    filename2 = 'outdata/' + op.mode + '/' + fileExt + '_' + date + '.txt'
    f = open(filename, 'r')
    g = open(filename2, 'r')
    plt.gcf()
    i = 0
    for line in f:
        separated = line.split(',')
        dat = [float(d) for d in separated[:-1]]    # Ignore carriage return
        tim = np.linspace(0, op.end_time, len(dat))
        plt.plot(tim, dat, label=g.readline().split(',')[0])
        i += 1
    f.close()
    g.close()
    if realData:
        x, y = extract_data(quantity)
        plt.plot(np.linspace(0, op.end_time, len(x)), y, label='Gauge data', marker='*', markevery=1, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel(quantity+' value')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.savefig('outdata/' + op.mode + '/' + fileExt + '_' + quantity + date + '.pdf', bbox_inches='tight')
    plt.clf()


def compare_timeseries(date, run, quantity, op=TohokuOptions()):
    assert quantity in ('P02', 'P06')
    approaches = ("FixedMesh", "HessianBased", "DWP", "DWR")

    # Get dates (if necessary)
    dates = {}
    now = datetime.datetime.now()
    today = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
    for approach in approaches:
        if date is None:
            try:
                dates[approach] = input("Date to use for {a:s} approach: ".format(a=approach))
            except:
                dates[approach] = today
        else:
            dates[approach] = date
    plt.gcf()
    for approach in approaches:
        try:
            filename = 'outdata/' + op.mode + '/' + approach + '_' + dates[approach] + quantity + '.txt'
            f = open(filename, 'r')
            for i in range(run):
                f.readline()
            separated = f.readline().split(',')
            dat = [float(d) for d in separated[:-1]]  # Ignore carriage return
            tim = np.linspace(0, op.end_time, len(dat))
            plt.plot(tim[::5], dat[::5], label=approach)
        except:
            pass
    plt.xlabel('Time (s)')
    plt.ylabel(quantity + ' value')
    plt.legend(loc=2)
    plt.savefig('outdata/' + op.mode + '/' + quantity + today + '_' +  str(run) + '.pdf', bbox_inches='tight')
    plt.clf()


def error_vs_elements(mode='Tohoku', noTinyMeshes=False, date=None, op=TohokuOptions()):
    now = datetime.datetime.now()
    today = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
    di = 'outdata/' + mode + '/'

    labels = ("Fixed mesh", "Hessian based", "DWP", "DWR")
    names = ("FixedMesh", "HessianBased", "DWP", "DWR")
    styles = {labels[0]: 's', labels[1]: '^', labels[2]: 'x', labels[3]: 'o'}
    err = {}
    nEls = {}
    tim = {}
    errorlabels = [r'Objective value $J(\textbf{q})=\int_{T_{\mathrm{start}}}^{T_{\mathrm{end}}}\int\int_A\eta(x,y,t)\,\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}t$']
    errornames = ['OF']
    if mode == 'Tohoku':
        errortypes = 2
        errorlabels.append('Relative total variation at gauge P02')
        errorlabels.append('Relative total variation at gauge P06')
        errornames.append('P02')
        errornames.append('P06')

    # Get dates (if necessary)
    dates = []
    for n in range(len(names)):
        if date is None:
            dates.append(input("Date to use for {a:s} approach: ".format(labels[n])))
        else:
            dates.append(date)

    for m in range(errortypes):
        for n in range(len(names)):
            try:
                av, errors, timing = read_errors(dates[n], names[n], mode)
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

        # PLot errors vs. elements
        for mesh in err:
            plt.gcf()
            plt.semilogx(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        if mode == 'Tohoku':
            plt.hlines(op.J, 4e3, 2e5, colors='k', linestyles='solid', label=r'681,616 elements')
            plt.axhspan(op.J-5e10, op.J+5e10, alpha=0.5, color='gray')

        plt.legend(loc=4)
        plt.xlabel(r'Mean element count')
        plt.ylabel(errorlabels[m])
        plt.savefig(di + errornames[m] + '-vs-elements' + today + '.pdf', bbox_inches='tight')
        plt.clf()

        # Plot errors vs. timings
        for mesh in err:
            plt.loglog(tim[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        if bootstrapping:
            if mode == 'Tohoku':
                plt.hlines(op.J, 8e1, 2e3, colors='k', linestyles='solid', label=r'681,616 elements')
                plt.axhspan(op.J-5e10, op.J+5e10, alpha=0.5, color='gray')
        plt.gcf()
        plt.legend(loc=4)
        plt.xlabel(r'CPU time (s)')
        plt.ylabel(errorlabels[m])
        plt.savefig(di + errornames[m] + '-vs-timings' + today + '.pdf', bbox_inches='tight')
        plt.clf()

        # Plot timings vs. elements
        if m == 0:
            for mesh in err:
                plt.loglog(nEls[mesh], tim[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
            plt.gcf()
            plt.legend(loc=2)
            plt.xlabel(r'Mean element count')
            plt.ylabel(r'CPU time (s)')
            plt.savefig(di + 'time-vs-elements' + today + '.pdf', bbox_inches='tight')
            plt.clf()
