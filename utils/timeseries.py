from firedrake import *
from firedrake.petsc import PETSc

import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt

from . import misc
from . import options as opt


def loadMesh(filename):
    """
    :arg filename: mesh filename to load from, including directory location.
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
    
    :arg gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :arg dirName: name of directory for locating HDF5 files, from the set {'fixedMesh', 'simpleAdapt', 'adjointBased'}
    :arg iEnd: final index.
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


def extractTimeseries(gauges, eta, t, current, v0, op=opt.Options()):
    """
    :arg gauges: list of gauge name strings, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :arg eta: Function to extract timeseries from.
    :arg t: current time.
    :arg current: dictionary containing timeseries data.
    :arg v0: dictionary containing initial gauge values.
    :param op: Options object holding parameter values.
    :return: dictionary containing all timeseries data so far.
    """
    for gauge in gauges:
        if gauge not in current.keys():
            current[gauge] = {}
            current[gauge][t] = 0.
        else:
            current[gauge][t] = float(eta.at(op.gaugeCoord(gauge))) - float(v0[gauge])
    return current


def saveTimeseries(gauge, data, name='test'):
    """
    :arg gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :arg data: timeseries data to save.
    :param name: name to give to timeseries.    
    """
    outfile = open('outdata/timeseries/' + gauge + name + '.txt', 'w+')
    for t in data[gauge].keys():
        outfile.write(str(t) + ' , ' + str(data[gauge][t]) + '\n')
    outfile.close()


def plotGauges(gauge, op=opt.Options()):
    """
    Read timeseries data for a particular gauge and calculate (L1, L2, L-infinity and TV) error norms.

    :arg gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param op: Options object holding parameter values.
    """
    T = op.Tend
    filename = 'outdata/timeseries/' + gauge + 'data_' + str(int(T/60)) + 'mins.txt'
    iEnd = sum(1 for line in open(filename))

    # Get plotting parameters
    labels = op.labels
    styles = op.styles
    stamps = op.stamps
    mM = [0, 0]     # Min/max for y-axis limits

    # Import data from the inversion analysis and interpolate
    measuredfile = open(filename, 'r')
    p = np.linspace(0, T/60, num=int(T))
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
        ts = []
        print('Timeseries to plot: ', mesh)
        name = stamps[mesh] + input("Version (hit enter to skip): ")
        if name != stamps[mesh]:
            infile = open('outdata/timeseries/' + gauge + name + '.txt', 'r')
            data = []
            i = 1
            for line in infile:
                pair = line.split(',')
                ts.append(float(pair[0])/60)    # Time in minutes
                val = float(pair[1])
                mVal = float(m(ts[-1]))
                if i == 1:
                    v0 = val - mVal
                    sStart = v0
                    mStart = mVal
                diff = val - v0 - mVal          # Approximation error
                absDiff = np.abs(diff)

                # Compute L1, L2, L-infinity errors and norms of gauge data using trapezium rule
                if i in (1, iEnd):
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
                if i == 2:
                    sign = (diff - diff_) / np.abs(diff - diff_)
                    mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
                elif i > 2:
                    sign_ = sign
                    mSign_ = mSign
                    sign = (diff - diff_) / np.abs(diff - diff_)
                    mSign = (mVal - mVal_) / np.abs(mVal - mVal_)
                    if sign != sign_:
                        error[3] += np.abs(diff_ - sStart)
                        sStart = diff_
                        if i == iEnd:
                            error[3] += np.abs(diff - diff_)
                    elif i == iEnd:
                        error[3] += np.abs(diff - sStart)
                    if mSign != mSign_:
                        norm[3] += np.abs(mVal_ - mStart)
                        mStart = mVal_
                        if i == iEnd:
                            norm[3] += np.abs(mVal - mVal_)
                    elif i == iEnd:
                        norm[3] += np.abs(mVal - mStart)
                diff_ = diff
                mVal_ = mVal
                if val < mM[0]:
                    mM[0] = val
                elif val > mM[1]:
                    mM[1] = val
                data.append(val)
                i += 1
            plt.plot(ts, data, label=mesh, marker=styles[mesh], markevery=5, linewidth=0.75)

            # Print errors to screen
            dt = ts[-1] - ts[-2]     # timestep TODO: needs altering to allow for t-adaptivity
            error[0] *= dt/2
            norm[0] *= dt/2
            error[1] = np.sqrt(error[1] * dt/2)
            norm[1] = np.sqrt(norm[1] * dt/2)
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


def errorVsElements(i=5):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('legend', fontsize='x-large')
    labels = ("Fixed mesh", "Hessian based", "Adjoint based", "Goal based", "Explicit estimator")
    styles = {labels[0]: 's', labels[1]: '^', labels[2]: 'x', labels[3]: 'o', labels[4]: '*'}
    err = {}
    nEls = {}
    tim = {}
    err[labels[0]] = [0.0046, 0.0067, 0.0022, 0.0083, 0.0034, 0.0026, 0.0005]
    nEls[labels[0]] = [6176, 8782, 11020, 16656, 20724, 33784, 52998]
    tim[labels[0]] = [11.7, 14.5, 16.8, 12.4, 17.8, 38.1, 98.5]
    err[labels[1]] = [0.0046, 0.0064, 0.0014, 0.0069, 0.0059]
    nEls[labels[1]] = [7036, 12540, 18638, 23699, 38610]
    tim[labels[1]] = [202.7, 420.8, 359.9, 619.6, 1295.0]
    err[labels[2]] = [0.0845, 0.0157, 0.0236, 0.0156]
    nEls[labels[2]] = [3410, 10088, 16610, 18277]
    tim[labels[2]] = [345.6, 562.4, 591.6, 1276.5]
    err[labels[3]] = [0.0209, 0.0027, 0.0020, 0.0010]
    nEls[labels[3]] = [3633, 12840, 27718, 43961]
    tim[labels[3]] = [895.2, 1873.9, 2655.5, 3198.2]
    err[labels[4]] = [0.0069, 0.0004, 0.0119]
    nEls[labels[4]] = [7963, 11051, 12841]
    tim[labels[4]] = [1459.3, 2245.1, 3306.4]

    # Plot errors
    cnt = 0
    for mesh in labels:
        plt.semilogy(nEls[mesh], err[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        cnt += 1
        if cnt == i:
            break
    plt.gcf()
    plt.legend(bbox_to_anchor=(0.6, 1.), loc=2)
    plt.xlabel(r'Mean element count')
    plt.ylabel(r'Relative error $\frac{|J(\textbf{q})-J(\textbf{q}_h)|}{|J(\textbf{q})|}$')
    plt.xlim([0, 55000])
    plt.ylim([0, 0.1])
    plt.savefig('outdata/errorPlots/errorVsElements' + str(i) + '.pdf', bbox_inches='tight')
    plt.show()

    # Plot timings
    cnt = 0
    for mesh in labels:
        plt.loglog(nEls[mesh], tim[mesh], label=mesh, marker=styles[mesh], linewidth=1.)
        cnt += 1
        if cnt == i:
            break
    plt.gcf()
    # plt.legend(bbox_to_anchor=(0.2, -0.1), loc=4)
    plt.xlabel(r'Mean element count')
    plt.ylabel(r'CPU time (s)')
    plt.xlim([0, 55000])
    plt.ylim([0, 4000])
    plt.savefig('outdata/errorPlots/timeVsElements' + str(i) + '.pdf', bbox_inches='tight')
    plt.show()
