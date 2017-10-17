import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt


def gaugeTimeseries(gauge, dat):
    """
    Store timeseries data for a particular gauge.
    
    :param gauge: gauge name string, from the set {'P02', 'P06', '801', '802', '803', '804', '806'}.
    :param dat: a list of data values of this gauge.
    :return: a file containing the timeseries data.
    """
    name = input('Enter a name for this time series (e.g. xcoarse): ')
    outfile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=name), 'w+')
    for i in range(len(dat)):
        outfile.write(str(dat[i]) + '\n')
    outfile.close()


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

    # Plot simulations and calculate error norms:
    x = np.linspace(0, 25, num=1501)
    for key in setup:
        val = []
        i = 0
        v0 = 0
        L1 = 0
        L2 = 0
        Linf = 0
        infile = open('timeseries/{y1}_{y2}.txt'.format(y1=gauge, y2=setup[key]), 'r')
        for line in infile:
            if i == 0:
                if error:
                    v0 = float(line) - float(m(x[i]))
                else:
                    v0 = float(line)
            if error:
                val.append(np.abs(float(line) - v0 - float(m(x[i]))))
                L1 += val[-1]
                L2 += val[-1] ** 2
                if val[-1] > Linf:
                    Linf = val[-1]
            else:
                val.append(float(line) - v0)
            i += 1
        infile.close()

        # Print norm values to screen:
        if error:
            print('\nL1 norm for ', setup[key], ' : ', L1 / 1501)
            print('L2 norm for ', setup[key], ' : ', np.sqrt(L2 / 1501))
            print('Linf norm for ', setup[key], ' : ', Linf)

        # Deal with special cases:
        if setup[key] in ('fine_nonlinear', 'fine_nonlinear_rotational',
                          'xcoarse_25mins', 'medium_25mins', 'fine_25mins',
                          'anisotropic_point85scaled_rm=30', 'goal-based_res4_fifthscaled',
                          'goal-based_better_version'):
            T = 25
        else:
            T = 60
        if log:
            plt.semilogy(np.linspace(0, T, len(val)), val, label=labels[key], marker=styles[key], markevery=60, linewidth=0.5)
        else:
            plt.plot(np.linspace(0, T, len(val)), val, label=labels[key], marker=styles[key], markevery=60, linewidth=0.5)

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
    if log:
        filename += 'log'
    else:
        filename += 'full'
    filename += '_gauge_timeseries_{y1}_{y2}'.format(y1=gauge, y2=prob)
    if error:
        filename += '_error'
    plt.savefig(filename + '.pdf', bbox_inches='tight')
