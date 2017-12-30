def indexString(index):
    """
    :param index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def cheatCodes(approach, default='goalBased'):
    """
    Enable skipping of sections of code using 'saved' and 'regen'.
    
    :param approach: user specified input.
    :return: approach to use and keys to skip sections.
    """
    approach = approach or default
    if approach == 'goalBased':
        getData = True
        getError = True
    elif approach == 'saved':
        approach = 'goalBased'
        getData = False
        getError = False
    elif approach == 'regen':
        approach = 'goalBased'
        getData = False
        getError = True
    elif approach == 'simpleAdapt':
        getData = False
        getError = False
    else:
        getData = True
        getError = False

    return approach, getData, getError


def printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer=False):
    """
    Print timings for the various sections of code.
    
    :param primalTimer: primal solver.
    :param dualTimer: dual solver.
    :param errorTimer: error estimation phase.
    :param adaptTimer: adaptive primal solver.
    :return: 
    """
    print("TIMINGS:")
    if bool(bootTimer):
        print("""
        Bootstrap run %5.3fs""" % (bootTimer))
    print("""
        Forward run   %5.3fs,
        Adjoint run   %5.3fs, 
        Error run     %5.3fs,
        Adaptive run   %5.3fs\n""" % (primalTimer, dualTimer, errorTimer, adaptTimer))