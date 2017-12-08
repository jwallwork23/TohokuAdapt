from firedrake import *

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from . import mesh as msh
from . import options as opt


def gaugeTimeseries(gauge, dirName, iEnd, op=opt.Options()):
    """
    Store timeseries data for a particular gauge and calculate (L1, L2, L-infinity) error norms.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :param iEnd: final index.
    :param op: Options object holding parameter values.
    """
    # Get solver parameters
    T = op.Tend / 60
    dt = op.dt
    rm = op.rm
    ndump = op.ndump
    # setup = op.plotDir
    # labels = op.labels
    # styles = op.styles
    numVals = int(T / (ndump * dt)) + 1

    # Import data from the inversion analysis and interpolate
    measuredfile = open('outdata/timeseries/' + gauge + 'data_' + str(int(T)) + 'mins.txt', 'r')
    p = np.linspace(0, T, num=numVals)
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.rc('legend', fontsize='x-large')
    # plt.clf()
    x = []
    y = []
    for line in measuredfile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    m = si.interp1d(x, y, kind=1)
    # plt.plot(p, m(p), label='Gauge measurement', linestyle='-', linewidth=2)

    name = input("Enter a name for this time series (e.g. 'meanEle=5767'): ")
    error = [0, 0, 0, 0]
    norm = [0, 0, 0, 0]

    # Write timeseries to file and calculate errors
    outfile = open('timeseries/' + gauge + '_' + name + '.txt', 'w+')
    val = []
    t = np.linspace(0, T, num=iEnd+1)
    for i in range(iEnd+1):
        indexStr = op.indexString(i)

        # Load mesh from file and set up Function to load into
        if not i % rm:
            elev_2d = Function(FunctionSpace(msh.loadMesh(dirName + 'hdf5/mesh_' + indexStr + '.h5'),
                                             op.space2, op.degree2))

        # Load data from HDF5 and get timeseries data
        with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
            loadElev.load(elev_2d, name='elev_2d')
            loadElev.close()
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

    # TODO separate out plotting script
    # # Plot timeseries data
    # for key in setup:
    #     data = gaugeTimeseries(gauge, op.plotDir[key], numVals)
    #     plt.plot(np.linspace(0, T, numVals), data,
    #              label=labels[key],
    #              marker=styles[key],
    #              markevery=60 / (ndump * dt),
    #              linewidth=0.5)
    # plt.xlim([0, T])
    # plt.gcf()
    # plt.legend(bbox_to_anchor=(1.13, 1.1), loc=2)
    # plt.ylim([-2, 5] if gauge == 'P02' else [-1, 5])  # TODO: remove this special casing using int and np.ceil
    # plt.xlabel(r'Time elapsed (mins)')
    # plt.ylabel(r'Free surface (m)')
    # plt.savefig('plots/timeseries/' + gauge + '.pdf', bbox_inches='tight')


def saveToDisk(f, g, dirName, index, filename='adjoint_'):
    """
    :param f: first function to save.
    :param g: second function to save.
    :param dirName: name of directory to save in.
    """
    with DumbCheckpoint(dirName + 'hdf5/' + filename + op.indexString(index), mode=FILE_CREATE) as chk:
        chk.store(f)
        chk.store(g)
        chk.close()

    # TODO: include option for any number of functions to save using *args
    # TODO: or possibly just remove this function.

if __name__ == '__main__':

    for gauge in ("P02", "P06"):
        dirName = input("dirName: ")
        iEnd = int(input("iEnd: "))
        useAdjoint = bool(input("useAdjoint?: "))
        op = opt.Options(vscale=0.4 if useAdjoint else 0.85,
                             rm=60 if useAdjoint else 30,
                             gradate=True if useAdjoint else False,
                             # gradate=False,
                             advect=False,
                             outputHessian=True,
                             coarseness=5,
                             gauges=True)
        gaugeTimeseries(gauge, dirName, iEnd, op=op)