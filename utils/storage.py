from firedrake import *

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from . import options


def indexString(index):
    """
    :param index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def gaugeTimeseries(gauge, dirName, iEnd):
    """
    Store timeseries data for a particular gauge and calculate (L1, L2, L-infinity) error norms.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :param iEnd: final HDF5 name string index.
    :return: a list containing the timeseries data.
    """
    op = options.Options()

    name = input("Enter a name for this time series (e.g. 'meanEle=5767'): ")
    dirName = 'plots/' + dirName + '/hdf5'
    error = [0, 0, 0, 0]
    norm = [0, 0, 0, 0]

    # Interpolate data from the inversion analysis and plot
    measuredfile = open('timeseries/' + gauge + '_measured_dat.txt', 'r')
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
    t = np.linspace(0, 25, num=int(iEnd + 1))
    for i in range(iEnd + 1):

        # Load data from HDF5 and get timeseries data TODO: how to define a function space if we do not know the mesh?
        with DumbCheckpoint(dirName + '/Elevation2d_' + indexString(i), mode=FILE_READ) as el:
            el.load(elev_2d, name='elev_2d')
        data = elev_2d.at(op.gaugeCoord(gauge))
        mVal = float(m(t[i]))
        if i == 0:
            v0 = data - mVal
            sStart = data
            mStart = mVal
        val.append(np.abs(data - v0 - mVal))

        # Compute L1, L2, L-infinity errors and norms of gauge data
        error[0] += val[-1]
        error[1] += val[-1] ** 2
        if val[-1] > error[2]:
            error[2] = val[-1]
        norm[0] += mVal
        norm[1] += mVal ** 2
        if mVal > norm[2]:
            norm[2] = mVal

        # Compute total variation of error and gauge data
        if i == 1:
            sign = (data - data_) / np.abs(data - data_)
            mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
        elif i > 1:
            sign_ = sign
            mSign_ = mSign
            sign = (data - data_) / np.abs(data - data_)
            mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
            if (sign != sign_) | (i == iEnd):
                error[3] += np.abs(data - sStart)
                sStart = data
            if (mSign != mSign_) | (i == iEnd):
                norm[3] += np.abs(mVal - mStart)
                mStart = mVal
        data_ = data
        mVal_ = mVal

        outfile.write(str(data) + '\n')
    outfile.close()

    # Print errors to screen
    error[0] /= (iEnd + 1)
    norm[0] /= (iEnd + 1)
    error[1] = np.sqrt(error[1] / (iEnd + 1))
    norm[1] = np.sqrt(norm[1] / (iEnd + 1))
    print("""Absolute L1 norm :         %5.2f Relative L1 norm :         %5.2f
             Absolute L2 norm :         %5.2f Relative L2 norm :         %5.2f
             Absolute L-infinity norm : %5.2f Relative L-infinity norm : %5.2f
             Absolute total variation : %5.2f Relative total variation : %5.2f""" %
          (error[0], error[0] / norm[0], error[1], error[1] / norm[1],
           error[2], error[2] / norm[2], error[3], error[3] / norm[3],))

    return val


def plotGauges(gauge):
    """
    Plot timeseries data on a single axis.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :return: a matplotlib plot of the corresponding gauge timeseries data.
    """
    op = options.Options()
    setup = op.plotDir
    labels = op.labels
    styles = op.styles
    measuredfile = open('timeseries/{y}_measured_dat_25mins.txt'.format(y=gauge), 'r')
    p = np.linspace(0, 25, num=1501)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    plt.clf()

    # Interpolate data from the inversion analysis and plot
    x = []
    y = []
    for line in measuredfile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    m = si.interp1d(x, y, kind=1)
    plt.plot(p, m(p), label='Gauge measurement', linestyle='-', linewidth=2)

    # Deal with special cases and plot timeseries data
    T = 25
    for key in setup:
        data = gaugeTimeseries(gauge, plotDir[key], 1501)
        plt.plot(np.linspace(0, T, 1501), data, label=labels[key], marker=styles[key], markevery=60, linewidth=0.5)
    plt.xlim([0, 25])
    plt.gcf()
    plt.legend(bbox_to_anchor=(1.13, 1.1), loc=2)
    plt.ylim([-2, 5] if gauge == 'P02' else [-1, 5])
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
    plt.savefig('plots/timeseries/' + gauge + '.pdf', bbox_inches='tight')
