from thetis import *

import scipy.interpolate as si
import matplotlib.pyplot as plt
import numpy as np
import datetime

from .options import Options


__all__ = ["readErrors", "extractSpline", "extractData", "errorVsElements", "__main__", "plotTimeseries",
           "compareTimeseries", "timeseriesDifference", "totalVariation", "gaugeTV", "integrateTimeseries"]


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')


def totalVariation(data):
    """
    :arg data: (one-dimensional) timeseries record.
    :return: total variation thereof.
    """
    TV = 0
    iStart = 0
    for i in range(len(data)):
        if i == 1:
            sign = (data[i] - data[i-1]) / np.abs(data[i] - data[i-1])
        elif i > 1:
            sign_ = sign
            sign = (data[i] - data[i - 1]) / np.abs(data[i] - data[i - 1])
            if sign != sign_:
                TV += np.abs(data[i-1] - data[iStart])
                iStart = i-1
                if i == len(data)-1:
                    TV += np.abs(data[i] - data[i-1])
            elif i == len(data)-1:
                TV += np.abs(data[i] - data[iStart])
    return TV


def gaugeTV(data, gauge="P02"):
    """
    :param data: timeseries to calculate error of.
    :param gauge: gauge considered.
    :return: total variation. 
    """
    N = len(data)
    spline = extractSpline(gauge)
    times = np.linspace(0., 25., N)
    errors = [data[i] - spline(times[i]) for i in range(N)]
    return totalVariation(errors) / totalVariation([spline(times[i]) for i in range(N)])


def readErrors(date, approach, mode='tohoku', bootstrapping=False):
    """
    :arg date: date simulation was run.
    :arg approach: mesh adaptive approach.
    :param mode: problem considered.
    :param bootstrapping: toggle use of bootstrapping.
    :return: mean element count, (some aspect of) error and CPU time.
    """
    filename = 'outdata/'+mode+'/'+approach+'_'+date
    textfile = open(filename+'.txt', 'r')
    if mode == 'model-verification':
        bootstrapping = True
    nEls = []
    err = {}
    tim = []
    i = 0
    for line in textfile:
        if mode == 'tohoku':
            av, rel, gP02, gP06, timing, J_h = line.split(',')
        elif mode == 'shallow-water':
            av, rel, timing, J_h = line.split(',')
        elif mode == 'rossby-wave':
            av, rel, peak, dis, spd, timing, J_h = line.split(',')
        if mode == 'model-verification':
            fixedMeshes = Options().meshSizes
            nEle, J_h, gP02, gP06, timing = line.split(',')[1:]
            nEls.append(fixedMeshes[i])
        else:
            nEls.append(int(av))
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


def extractData(gauge):
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
        measuredFile = open('outdata/rossby-wave/analytic_Integrand.txt', 'r')
        dat = measuredFile.readline()
        xy = dat.split(",")
        measuredFile.close()

        return range(len(xy)-1), [float(i) for i in xy[:-1]]


def plotTimeseries(fileExt, date, quantity='Integrand', realData=False, op=Options()):
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
        tim = np.linspace(0, op.Tend, len(dat))
        plt.plot(tim[::5], dat[::5], label=g.readline().split(',')[0])
        i += 1
    f.close()
    g.close()
    if realData:
        if (op.mode == 'tohoku' and quantity in ('P02', 'P06')) or op.mode == 'rossby-wave':
            x, y = extractData(quantity)
            me = 10 if op.mode == 'rossby-wave' else 1
            plt.plot(np.linspace(0, op.Tend, len(x)), y, label='Gauge data', marker='*', markevery=me, color='black')
    plt.xlabel('Time (s)')
    plt.ylabel(quantity+' value')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    plt.savefig('outdata/' + op.mode + '/' + fileExt + '_' + quantity + date + '.pdf', bbox_inches='tight')
    plt.clf()


def timeseriesDifference(fileExt1, date1, fileExt2, date2, quantity='Integrand', op=Options()):
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
            errs.append(totalVariation(np.asarray(dat1) - np.asarray(dat2)))
        except:
            pass
    return ['%.4e' % i for i in errs]


def integrateTimeseries(fileExt, date, op=Options()):
    if date is None:
        date = ''
    filename = 'outdata/' + op.mode + '/' + fileExt + '_' + date + 'Integrand.txt'
    f = open(filename, 'r')
    integrals = []
    for line in f:
        separated = line.split(',')
        dat = [float(d) for d in separated[:-1]]
        I = 0
        # dt = op.Tend / len(dat)
        dt = op.dt
        for i in range(1, len(dat)):
            I += 0.5 * (dat[i] + dat[i-1]) * dt
        integrals.append(I)
    return ['%.4e' % i for i in integrals]


def compareTimeseries(date, run, quantity='Integrand', op=Options()):
    assert quantity in ('Integrand', 'P02', 'P06')
    approaches = ("fixedMesh", "hessianBased", "DWP", "DWR")

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


def errorVsElements(mode='tohoku', bootstrapping=False, noTinyMeshes=False, date=None):
    now = datetime.datetime.now()
    today = str(now.day) + '-' + str(now.month) + '-' + str(now.year % 2000)
    di = 'outdata/' + mode + '/'

    if mode == 'model-verification':
        labels = ("Non-rotational", "Rotational")
        names = ("nonlinear=False_", "nonlinear=True_")
    else:
        labels = ("Fixed mesh", "Hessian based", "DWP", "DWR", "Higher order DWR", "Refined DWR")
        names = ("fixedMesh", "hessianBased", "DWP", "DWR", "DWR_ho", "DWR_r")
    styles = {labels[0]: 's', labels[1]: '^'}
    if mode != 'model-verification':
        styles[labels[2]] = 'x'
        styles[labels[3]] = 'o'
        styles[labels[4]] = '*'
        styles[labels[5]] = 'h'
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
            dates.append(input("Date to use for %s approach: " % labels[n]))
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

        # PLot errors vs. elements
        for mesh in err:
            plt.gcf()
            if bootstrapping:
                plt.semilogx(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
                plt.legend(loc=1 if errornames[m] in ('P02', 'P06') else 4)
            else:
                plt.loglog(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
                plt.legend(loc=1)
            plt.xlabel(r'Mean element count')
            plt.ylabel(errorlabels[m])
            plt.savefig(di + errornames[m] + 'VsElements' + today + '.pdf', bbox_inches='tight')
            plt.clf()

        # Plot errors vs. timings
        for mesh in err:
            plt.loglog(tim[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        plt.gcf()
        plt.legend(loc=2)
        plt.xlabel(r'CPU time (s)')
        plt.ylabel(errorlabels[m])
        plt.savefig(di + errorlabels[m] + 'VsTimings' + today + '.pdf', bbox_inches='tight')
        plt.clf()

        # Plot timings vs. elements
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
            plt.savefig(di + 'timeVsElements' + today + '.pdf', bbox_inches='tight')
            plt.clf()
