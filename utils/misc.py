def indexString(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def cheatCodes(approach, default='goalBased'):
    """
    Enable skipping of sections of code using 'saved' and 'regen'.
    
    :arg approach: user specified input.
    :return: approach to use and keys to skip sections.
    """
    approach = approach or default
    if approach == 'goalBased':
        getData = True
        getError = True
        useAdjoint = True
    elif approach == 'simpleAdapt':
        getData = False
        getError = False
        useAdjoint = False
    elif approach == 'explicit':
        getData = True
        getError = True
        useAdjoint = False
    elif approach == 'adjointBased':
        getData = True
        getError = True
        useAdjoint = True
    elif approach == 'saved':
        approach = input("Choose error estimator: 'explicit', 'adjointBased' or 'goalBased': ") or 'goalBased'
        getData = False
        getError = False
        useAdjoint = True
    elif approach == 'regen':
        approach = input("Choose error estimator: 'explicit', 'adjointBased' or 'goalBased': ") or 'goalBased'
        getData = False
        getError = True
        useAdjoint = approach in ('adjointBased', 'goalBased')
    else:
        approach = 'fixedMesh'
        getData = True
        getError = False
        useAdjoint = False

    return approach, getData, getError, useAdjoint


def printTimings(primalTimer, dualTimer, errorTimer, adaptTimer, bootTimer=False):
    """
    Print timings for the various sections of code.
    
    :arg primalTimer: primal solver.
    :arg dualTimer: dual solver.
    :arg errorTimer: error estimation phase.
    :arg adaptTimer: adaptive primal solver.
    :arg bootTimer: bootstrapping routine.
    :return: 
    """
    print("TIMINGS:")
    if bool(bootTimer):
        print("""
        Bootstrap run %5.3fs""" % (bootTimer))
    else:
        bootTimer = 0
    print("""
        Forward run   %5.3fs,
        Adjoint run   %5.3fs, 
        Error run     %5.3fs,
        Adaptive run  %5.3fs
        Total         %5.3fs\n""" % (primalTimer, dualTimer, errorTimer, adaptTimer,
                                     primalTimer + dualTimer + errorTimer + adaptTimer + bootTimer))