from firedrake import *
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from . import error
from . import options


def indexString(index):
    """
    :param index: integer form of index.
    :return: five-digit string form of index.
    """
    indexStr = str(index)
    for i in range(5 - len(indexStr)):
        indexStr = '0' + indexStr
    return indexStr


def gaugeTimeseries(gauge, dirName, iEnd):
    """
    Store timeseries data for a particular gauge and calculate (L1, L2, L-infinity) error norms.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :param iEnd: final HDF5 name string index.
    :return: a file containing the timeseries data.
    """
    op = options.Options()
    name = input("Enter a name for this time series (e.g. 'meanEle=5767'): ")
    dirName = 'plots/' + dirName + '/hdf5'
    error = [0, 0, 0, 0]

    # Interpolate data from the inversion analysis and plot
    measuredfile = open('timeseries/{}_measured_dat.txt'.format(gauge), 'r')
    x = []
    y = []
    for line in measuredfile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    m = si.interp1d(x, y, kind=1)

    # Write timeseries to file and calculate errors
    outfile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=name), 'w+')
    val = []
    t = np.linspace(0, 25, num=1501)    # TODO: generalise number 1501
    for i in range(iEnd + 1):
        # TODO: how to define a function space if we do not know the mesh?
        with DumbCheckpoint(dirName + '/Elevation2d_' + indexString(i), mode=FILE_READ) as el:
            el.load(elev_2d, name='elev_2d')
        data = elev_2d.at(op.gaugeCoord(gauge))
        if i == 0:
            v0 = data - float(m(t[i]))
        val.append(np.abs(data - v0 - float(m(t[i]))))
        error[0] += val[-1]
        error[1] += val[-1] ** 2
        if val[-1] > error[2]:
            error[2] = val[-1]
        outfile.write(str(data) + '\n')
    outfile.close()
    error[3] = err.totalVariation(val)

    # Print errors to screen
    print('L1 norm : %5.2f, L2 norm : %5.2f, Linf norm : %5.2f, Total variation : %5.2f',
          (error[0] / 1501, np.sqrt(error[1] / 1501), error[2], error[3]))


def plotGauges(gauge, prob='comparison', log=False, error=False):
    """
    Plot timeseries data on a single axis.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param prob: problem type name string, corresponding to either 'verification' or 'comparison'.
    :param log: specify whether or not to use a logarithmic scale on the y-axis.
    :param error: make an error plot.
    :return: a matplotlib plot of the corresponding gauge timeseries data.
    """
    if prob == 'comparison':
        setup = {1: 'xcoarse_25mins',                       # Fixed with 3,126 vertices
                 2: 'medium_25mins',                        # Fixed with 25,976 vertices
                 3: 'fine_25mins',                          # Fixed with 97,343 vertices
                 4: 'anisotropic_point85scaled_rm=30',      # 'Simple adaptive': numVer = 0.85, rm=30, N1=3126
                 # 5: 'goal-based_res4_fifthscaled'}          # Goal-based adaptive: numVer = 0.2, rm=60, N1=7194
                 5: 'goal-based_better_version'}
        labels = {1: 'Coarse mesh',
                  2: 'Medium mesh',
                  3: 'Fine mesh',
                  4: 'Simple adaptive',
                  5: 'Goal based'}
        measuredfile = open('timeseries/{y}_measured_dat_25mins.txt'.format(y=gauge), 'r')
        p = np.linspace(0, 25, num=1501)
    else:
        setup = {1: 'fine',
                 2: 'fine_rotational',
                 3: 'fine_nonlinear',
                 4: 'fine_nonlinear_rotational'}
        labels = {1: 'Linear, non-rotational equations',
                  2: 'Linear, rotational equations',
                  3: 'Nonlinear, non-rotational equations',
                  4: 'Nonlinear, rotational equations'}
        measuredfile = open('timeseries/{y}_measured_dat.txt'.format(y=gauge), 'r')
        p = np.linspace(0, 60, num=3601)
    styles = {1: 's', 2: '^', 3: 'x', 4: 'o', 5: '*'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    plt.clf()

    # Interpolate data from the inversion analysis and plot:
    x = []
    y = []
    for line in measuredfile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    m = si.interp1d(x, y, kind=1)
    if not error:
        if log:
            plt.semilogy(p, m(p), label='Gauge measurement', linestyle='-', linewidth=2)
        else:
            plt.plot(p, m(p), label='Gauge measurement', linestyle='-', linewidth=2)

        # Deal with special cases:
        T = 25 if setup[key] in ('fine_nonlinear', 'fine_nonlinear_rotational',
                                 'xcoarse_25mins', 'medium_25mins', 'fine_25mins',
                                 'anisotropic_point85scaled_rm=30', 'goal-based_res4_fifthscaled',
                                 'goal-based_better_version') else T = 60
        if log:
            plt.semilogy(np.linspace(0, T, len(val)), val, label=labels[key], marker=styles[key], markevery=60,
                         linewidth=0.5)
        else:
            plt.plot(np.linspace(0, T, len(val)), val, label=labels[key], marker=styles[key], markevery=60,
                     linewidth=0.5)

        if prob == 'comparison':
            plt.xlim([0, 25])
        else:
            plt.xlim([0, 60])
    plt.gcf()
    if error:
        plt.legend(loc=2, facecolor='white') # 'upper right' == 1 and anticlockwise
    else:
        if prob == 'comparison':
            plt.legend(bbox_to_anchor=(1.13, 1.1), loc=1)
        else:
            plt.legend(bbox_to_anchor=(1.1, 1), loc=1)
    if log:
        plt.ylim([10 ** -3, 10 ** 2])
    elif gauge == 'P02':
        if error:
            plt.ylim([0, 1])
        else:
            plt.ylim([-2, 5])
    else:
        if error:
            plt.ylim([0, 1])
        else:
            plt.ylim([-1, 5])
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')

    # Set filename and save:
    filename = 'plots/tsunami_outputs/screenshots/'
    filename += 'log' if log else 'full'
    filename += '_gauge_timeseries_{y1}_{y2}'.format(y1=gauge, y2=prob)
    if error:
        filename += '_error'
    plt.savefig(filename + '.pdf', bbox_inches='tight')
