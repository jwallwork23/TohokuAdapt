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


def gaugeTimeseries(gauge, dirName, iEnd, op=opt.Options(), output=False, name='test', adaptive=True):
    """
    Extract timeseries data from HDF5 for a particular gauge.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :param iEnd: final index.
    :param op: Options object holding parameter values.
    :param output: toggle printing timeseries values to screen.
    :param name: name to give to timeseries.
    :param adaptive: adaptive mesh?
    """

    # Import data from the inversion analysis and get first value
    measuredfile = open('outdata/timeseries/' + gauge + 'data_' + str(int(op.Tend/60)) + 'mins.txt', 'r')
    m0 = float(measuredfile.readline().split()[0])

    # Write timeseries to file and calculate errors
    outfile = open('outdata/timeseries/' + gauge + name + '.txt', 'w+')
    # for i in range(0, iEnd+1, op.ndump):
    for i in range(0, iEnd+1):
        indexStr = misc.indexString(i)

        # Load mesh from file and set up Function to load into
        if ((not i % op.rm) & adaptive) | (i == 0):
            mesh = Mesh('resources/meshes/TohokuCoarse.msh')
            elev_2d = Function(FunctionSpace(mesh, op.space2, op.degree2))

        # Load data from HDF5 and get timeseries data
        with DumbCheckpoint(dirName + 'hdf5/Elevation2d_' + indexStr, mode=FILE_READ) as loadElev:
            loadElev.load(elev_2d, name='elev_2d')
            loadElev.close()

        # Interpolate elevation value at gauge location
        data = elev_2d.at(op.gaugeCoord(gauge))
        if i == 0:
            v0 = data - float(m0)
        outfile.write(str(data - v0) + '\n')
    outfile.close()


def extractTimeseries(gauges, eta, current, v0, op=opt.Options()):
    """
    :param gauges: list of gauge name strings, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param eta: Function to extract timeseries from.
    :param current: dictionary containing previous timeseries data.
    :param v0: dictionary containing initial gauge values.
    :param op: Options object holding parameter values.
    :return: dictionary containing all timeseries data so far.
    """

    for gauge in gauges:
        if gauge not in current.keys():
            current[gauge] = []
            current[gauge].append(0.)
        else:
            current[gauge].append(float(eta.at(op.gaugeCoord(gauge)) - v0[gauge]))
    return current


def saveTimeseries(gauge, data, name='test'):
    """
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param data: Timeseries data to save.
    :param name: name to give to timeseries.    
    """
    outfile = open('outdata/timeseries/' + gauge + name + '.txt', 'w+')
    for i in range(len(data)):
        outfile.write(str(data[i]) + '\n')
    outfile.close()


def plotGauges(gauge, iEnd, op=opt.Options()):
    """
    Read timeseries data for a particular gauge and calculate (L1, L2, L-infinity and TV) error norms.

    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param iEnd: final index.
    :param op: Options object holding parameter values.
    """

    # Get solver parameters
    T = op.Tend
    dt = op.dt
    ndump = op.ndump
    numStrips = int(T / (ndump * dt))
    numVals = numStrips + 1

    # Get plotting parameters
    labels = op.labels
    styles = op.styles
    stamps = op.stamps
    mM = [0, 0]     # Min/max for y-axis limits

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
        val = float(xy[1])
        y.append(val)
        if val < mM[0]:
            mM[0] = val
        elif val > mM[1]:
            mM[1] = val
    m = si.interp1d(x, y, kind=1)
    plt.plot(p, m(p), label='Gauge measurement', linestyle='-', linewidth=2)

    # Plot timeseries data
    print('\nGauge to plot: ' + gauge + '\n')
    for mesh in labels:
        error = [0, 0, 0, 0]
        norm = [0, 0, 0, 0]
        t = np.linspace(0, T / 60, num=iEnd + 1)
        print('Timeseries to plot: ', mesh)
        name = stamps[mesh] + input("Version (hit enter to skip): ")
        if name != stamps[mesh]:
            infile = open('outdata/timeseries/' + gauge + name + '.txt', 'r')
            data = []
            i = 0
            for line in infile:
                val = float(line)
                mVal = float(m(t[i]))
                if i == 0:
                    v0 = val - mVal
                    sStart = v0
                    mStart = mVal
                diff = val - v0 - mVal
                absDiff = np.abs(diff)

                # Compute L1, L2, L-infinity errors and norms of gauge data using trapezium rule
                if i in (0, iEnd):
                    error[0] += absDiff
                    error[1] += absDiff ** 2
                    norm[0] += np.abs(mVal)
                    norm[1] += mVal ** 2
                else:
                    error[0] += 2 * absDiff
                    error[1] += 2 * absDiff ** 2
                    norm[0] += 2 * np.abs(mVal)
                    norm[1] += 2 * mVal ** 2
                if absDiff > error[2]:
                    error[2] = absDiff
                if mVal > norm[2]:
                    norm[2] = mVal

                # Compute total variation of error and gauge data
                if i == ndump:
                    sign = (diff - diff_) / np.abs(diff - diff_)
                    mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
                elif i > ndump:
                    sign_ = sign
                    mSign_ = mSign
                    sign = (diff - diff_) / np.abs(diff - diff_)
                    mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
                    if (sign != sign_) | (i == iEnd):
                        error[3] += np.abs(diff - sStart)
                        sStart = diff
                    if (mSign != mSign_) | (i == iEnd):
                        norm[3] += np.abs(mVal - mStart)
                        mStart = mVal
                diff_ = diff
                mVal_ = mVal
                if val < mM[0]:
                    mM[0] = val
                elif val > mM[1]:
                    mM[1] = val
                data.append(val)
                # i += ndump
                i += 1
            plt.plot(np.linspace(0, T/60, numVals), data,
                     label=mesh,
                     marker=styles[mesh],
                     markevery=10,
                     linewidth=0.75)

            # Print errors to screen
            h = 1 / numStrips
            error[0] *= h / 2
            norm[0] *= h / 2
            error[1] = np.sqrt(error[1] * h / 2)
            norm[1] = np.sqrt(norm[1] * h / 2)
            print('\n' + gauge + """
Absolute L1 norm :         %6.3f Relative L1 norm :         %6.3f
Absolute L2 norm :         %6.3f Relative L2 norm :         %6.3f
Absolute L-infinity norm : %6.3f Relative L-infinity norm : %6.3f
Absolute total variation : %6.3f Relative total variation : %6.3f""" %
                  (error[0], error[0] / norm[0], error[1], error[1] / norm[1],
                   error[2], error[2] / norm[2], error[3], error[3] / norm[3],))

    plt.xlim([0, T/60])
    plt.gcf()
    plt.legend(bbox_to_anchor=(0.6, 1.), loc=2)
    plt.ylim([np.floor(mM[0]), np.ceil(mM[1])])
    plt.xlabel(r'Time elapsed (minutes)')
    plt.ylabel(r'Free surface displacement (m)')
    plt.savefig('outdata/timeseries/plots/' + gauge + '.pdf', bbox_inches='tight')
    plt.show()


def errorVsTime(op=opt.Options()):
    """
    :param op: Options object holding parameter values. 
    :return: plot of relative total variation versus simulation run time.
    """

    # Get plotting parameters
    # labels = op.labels
    labels = ("Coarse mesh", "Simple adaptive", "Goal based")
    styles = op.styles
    t = []
    e = []

    for mesh in labels:
        for gauge in ("P02", "P06"):
            print('Datum to plot: ', mesh, gauge)
            t.append(input("Run time to solution: "))
            e.append(input("Corresponding error: "))
            plt.plot(t, e, label=mesh, marker=styles[mesh], linewidth=0.)

    plt.gcf()
    plt.legend(bbox_to_anchor=(1.01, 1.), loc=2)
    plt.xlabel(r'Time elapsed (s)')
    plt.ylabel(r'Relative total variation')
    plt.xlim([0, 50000])
    plt.ylim([0, 1])
    plt.savefig('outdata/errorPlots/errorVsTime.pdf', bbox_inches='tight')
    plt.show()

