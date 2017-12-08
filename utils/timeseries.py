from firedrake import *
from firedrake.petsc import PETSc

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from . import misc
from . import options as opt


def loadMesh(filename):
    """
    :param filename: mesh filename to load from, including directory location.
    :return: Mesh, as loaded from HDF5.
    """
    filename += '.h5'
    # print("### loadMesh DEBUG: attempting to load " + filename)
    plex = PETSc.DMPlex().create()
    plex.createFromFile(filename)
    return Mesh(plex)


def gaugeTimeseries(gauge, dirName, iEnd, op=opt.Options(), output=False, name='test'):
    """
    Store timeseries data for a particular gauge and calculate (L1, L2, L-infinity) error norms.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :param iEnd: final index.
    :param op: Options object holding parameter values.
    :param output: toggle printing timeseries values to screen.
    """

    # Get solver parameters
    T = op.Tend / 60
    dt = op.dt
    ndump = op.ndump
    rm = op.rm

    # Import data from the inversion analysis and interpolate
    measuredfile = open('outdata/timeseries/' + gauge + 'data_' + str(int(T)) + 'mins.txt', 'r')
    x = []
    y = []
    for line in measuredfile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    m = si.interp1d(x, y, kind=1)

    error = [0, 0, 0, 0]
    norm = [0, 0, 0, 0]

    # Write timeseries to file and calculate errors
    outfile = open('outdata/timeseries/' + gauge + name + '.txt', 'w+')
    val = []
    t = np.linspace(0, T, num=iEnd+1)
    for i in range(0, iEnd+1, ndump):
        indexStr = misc.indexString(i)

        # Load mesh from file and set up Function to load into
        if not i % rm:
            elev_2d = Function(FunctionSpace(loadMesh(dirName + 'hdf5/mesh_' + indexStr), op.space2, op.degree2))

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
        toPlot = data - v0
        val.append(np.abs(data - mVal))
        if output:
            print('Time %.2f mins : %.4fm' % (i/60, toPlot))

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
        if i == ndump:
            sign = (data - data_) / np.abs(data - data_)
            mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
        elif i > ndump:
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

        outfile.write(str(toPlot) + '\n')
    outfile.close()

    # Print errors to screen
    error[0] /= (iEnd + 1)
    norm[0] /= (iEnd + 1)
    error[1] = np.sqrt(error[1] / (iEnd + 1))
    norm[1] = np.sqrt(norm[1] / (iEnd + 1))
    print("""
Absolute L1 norm :         %6.3f Relative L1 norm :         %6.3f
Absolute L2 norm :         %6.3f Relative L2 norm :         %6.3f
Absolute L-infinity norm : %6.3f Relative L-infinity norm : %6.3f
Absolute total variation : %6.3f Relative total variation : %6.3f""" %
          (error[0], error[0] / norm[0], error[1], error[1] / norm[1],
           error[2], error[2] / norm[2], error[3], error[3] / norm[3],))


def plotGauges(gauge, dirName, iEnd, op=opt.Options()):
    """
    Store timeseries data for a particular gauge and calculate (L1, L2, L-infinity) error norms.

    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :param iEnd: final index.
    :param op: Options object holding parameter values.
    """

    # Get solver parameters
    T = op.Tend
    dt = op.dt
    ndump = op.ndump
    numVals = int(T / (ndump * dt)) + 1

    # Get plotting parameters
    labels = op.labels
    styles = op.styles

    # Import data from the inversion analysis and interpolate
    measuredfile = open('outdata/timeseries/' + gauge + 'data_' + str(int(T/60)) + 'mins.txt', 'r')
    p = np.linspace(0, T/60, num=numVals)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    plt.clf()
    x = []
    y = []
    for line in measuredfile:
        xy = line.split()
        x.append(float(xy[0]))
        y.append(float(xy[1]))
    m = si.interp1d(x, y, kind=1)
    plt.plot(p, m(p), label='Gauge measurement', linestyle='-', linewidth=2)

    # Plot timeseries data
    for mesh in labels:
        print('Timeseries to plot: ' + mesh)
        name = input("Filename (hit enter to skip): ")
        if name != '':
            infile = open('outdata/timeseries/' + gauge + name + '.txt', 'r')
            data = []
            for line in infile:
                data.append(float(line))
            plt.plot(np.linspace(0, T/60, numVals), data,
                     label=mesh,
                     marker=styles[mesh],
                     markevery=ndump,
                     linewidth=0.5)
    plt.xlim([0, T/60])
    plt.gcf()
    plt.legend(bbox_to_anchor=(1.13, 1.1), loc=2)
    plt.ylim([-2, 5] if gauge == 'P02' else [-1, 5])  # TODO: remove this special casing using int and np.ceil
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
    plt.savefig('outdata/timeseries/plots/' + gauge + '.pdf', bbox_inches='tight')
    plt.show()
