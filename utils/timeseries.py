from thetis import *

import scipy.interpolate as si
import matplotlib.pyplot as plt
import numpy as np
import datetime

from .options import TohokuOptions


__all__ = ["error_vs_elements", "plot_timeseries", "compare_timeseries", "timeseries_difference", "integrate_timeseries"]


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')


def read_errors(date, approach, mode='Tohoku'):
    """
    :arg date: date simulation was run.
    :arg approach: mesh adaptive approach.
    :param mode: problem considered.
    :return: mean element count, (some aspect of) error and CPU time.
    """
    filename = 'outdata/'+mode+'/'+approach+'_'+date
    textfile = open(filename+'.txt', 'r')
    nEls = []
    err = {}
    tim = []
    i = 0
    for line in textfile:
        if mode == 'Tohoku':
            av, rel, gP02, gP06, timing, J_h = line.split(',')
        elif mode == 'GaussianTest':
            quantities = line.split(',')
            av = quantities[0]
            rel = quantities[1]
            timing = quantities[-2]
            J_h = quantities[-1]
        elif mode == 'RossbyWave':
            # av, rel, peak, dis, spd, timing, J_h = line.split(',')   # TODO: update for generalised framework
            quantities = line.split(',')
            av = quantities[0]
            rel = quantities[1]
            timing = quantities[-2]
            J_h = quantities[-1]
        else:
            nEls.append(int(av))
        if mode == 'Tohoku':
            err[i] = [float(rel), float(gP02), float(gP06)]
        elif mode == 'GaussianTest':
            err[i] = [float(rel)]
        elif mode == 'RossbyWave':
            # err[i] = [float(rel), float(peak), float(dis), float(spd)]
            err[i] = [float(rel)]
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
    elif gauge == "Integrand":
        measuredFile = open('outdata/RossbyWave/analytic_'+gauge+'.txt', 'r')
        dat = measuredFile.readline()
        xy = dat.split(",")
        measuredFile.close()

        return range(len(xy)-1), [float(i) for i in xy[:-1]]


def plot_timeseries(fileExt, date, quantity='Integrand', realData=False, op=TohokuOptions()):
    assert quantity in ('Integrand', 'P02', 'P06')
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
        if op.mode != 'GaussianTest':
            plt.plot(tim[::5], dat[::5], label=g.readline().split(',')[0])
        else:
            plt.plot(tim, dat, label=g.readline().split(',')[0])
        i += 1
    f.close()
    g.close()
    if realData:
        if (op.mode == 'Tohoku' and quantity in ('P02', 'P06')) or op.mode == 'RossbyWave':
            x, y = extract_data(quantity)
            me = 10 if op.mode == 'RossbyWave' else 1
            plt.plot(np.linspace(0, op.end_time, len(x)), y, label='Gauge data', marker='*', markevery=me, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel(quantity+' value')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.savefig('outdata/' + op.mode + '/' + fileExt + '_' + quantity + date + '.pdf', bbox_inches='tight')
    plt.clf()


def timeseries_difference(fileExt1, date1, fileExt2, date2, quantity='Integrand', op=TohokuOptions()):
    assert quantity in ('Integrand', 'P02', 'P06')
    filename1 = 'outdata/' + op.mode + '/' + fileExt1 + '_' + date1 + quantity + '.txt'
    filename2 = 'outdata/' + op.mode + '/' + fileExt2 + '_' + date2 + quantity + '.txt'
    f1 = open(filename1, 'r')
    f2 = open(filename2, 'r')
    errs = []
    for line in f1:
        separated = line.split(',')
        dat1 = [float(d) for d in separated[:-1]]
        separated = f2.readline().split(',')
        dat2 = [float(d) for d in separated[:-1]]
        try:
            assert np.shape(dat1) == np.shape(dat2)
            errs.append(total_variation(np.asarray(dat1) - np.asarray(dat2)))
        except:
            pass
    return ['%.4e' % i for i in errs]


def integrate_timeseries(fileExt, date, op=TohokuOptions()):
    if date is None:
        date = ''
    filename = 'outdata/' + op.mode + '/' + fileExt + '_' + date + 'Integrand.txt'
    f = open(filename, 'r')
    integrals = []
    for line in f:
        separated = line.split(',')
        # for j in range(len(separated), 0, -1):
        #     if j % (op.timesteps_per_remesh+2) in (op.timesteps_per_remesh,op.timesteps_per_remesh+1):
        #         del separated[j]
        dat = [float(d) for d in separated[:-1]]
        print("#### DEBUG: Number of timesteps stored = {n:d}".format(n=len(dat)))
        I = 0
        dt = op.timestep
        for i in range(1, len(dat)):
            I += 0.5 * (dat[i] + dat[i-1]) * dt
        integrals.append(I)
    return ['%.4e' % i for i in integrals]


def compare_timeseries(date, run, quantity='Integrand', op=TohokuOptions()):
    assert quantity in ('Integrand', 'P02', 'P06')
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


def error_vs_elements(mode='Tohoku',
                    bootstrapping=False,
                    noTinyMeshes=False,
                    date=None,
                    op=TohokuOptions()):
    now = datetime.datetime.now()
    today = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
    di = 'outdata/' + mode + '/'

    if mode == 'model-verification':
        labels = ("Non-rotational", "f-plane", "beta-plane", "Full")
        names = ("rotational=off", "rotational=f", "rotational=beta", "rotational=sin")
    else:
        labels = ("Fixed mesh", "Hessian based", "DWP", "DWR", "Higher order DWR", "Refined DWR")
        names = ("FixedMesh", "HessianBased", "DWP", "DWR", "DWR_ho", "DWR_r")
    styles = {labels[0]: 's', labels[1]: '^', labels[2]: 'x', labels[3]: 'o'}
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
    elif mode == 'GaussianTest':
        errortypes = 1
    elif mode == 'RossbyWave':
        # errortypes = 4
        # errorlabels.append('Relative error in solition peak')
        # errorlabels.append('Relative error in distance travelled')
        # errorlabels.append('Relative error in phase speed')
        # errornames.append('peak')
        # errornames.append('dis')
        # errornames.append('spd')
        errortypes = 1

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
                av, errors, timing = read_errors(dates[n], names[n], mode, bootstrapping=bootstrapping)
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
            if bootstrapping:
                plt.semilogx(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
            else:
                plt.loglog(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        if bootstrapping:
            if mode == 'RossbyWave':
                plt.hlines(op.J, 1e2, 1e6, colors='k', linestyles='solid', label=r'$1^{st}$ order asymptotic solution')
            elif mode == 'Tohoku':
                plt.hlines(op.J, 4e3, 2e5, colors='k', linestyles='solid', label=r'681,616 elements')
                plt.axhspan(op.J-5e10, op.J+5e10, alpha=0.5, color='gray')

        plt.legend(loc=4)
        plt.xlabel(r'Mean element count')
        plt.ylabel(errorlabels[m])
        plt.savefig(di + errornames[m] + 'VsElements' + today + '.pdf', bbox_inches='tight')
        plt.clf()

        # Plot errors vs. timings
        for mesh in err:
            plt.loglog(tim[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        if bootstrapping:
            if mode == 'RossbyWave':
                plt.hlines(op.J, 1e2, 1e6, colors='k', linestyles='solid', label=r'$1^{st}$ order asymptotic solution')
            elif mode == 'Tohoku':
                plt.hlines(op.J, 8e1, 2e3, colors='k', linestyles='solid', label=r'681,616 elements')
                plt.axhspan(op.J-5e10, op.J+5e10, alpha=0.5, color='gray')
        plt.gcf()
        plt.legend(loc=4)
        plt.xlabel(r'CPU time (s)')
        plt.ylabel(errorlabels[m])
        plt.savefig(di + errornames[m] + 'VsTimings' + today + '.pdf', bbox_inches='tight')
        plt.clf()

        # Plot timings vs. elements
        if m == 0:
            for mesh in err:
                plt.loglog(nEls[mesh], tim[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
            plt.gcf()
            plt.legend(loc=2)
            plt.xlabel(r'Mean element count')
            plt.ylabel(r'CPU time (s)')
            plt.savefig(di + 'timeVsElements' + today + '.pdf', bbox_inches='tight')
            plt.clf()
