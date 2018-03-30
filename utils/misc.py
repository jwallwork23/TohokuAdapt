__all__ = ["indexString", "cheatCodes", "printTimings", "getMax"]


def indexString(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def cheatCodes(approach, default='DWR'):
    """
    Enable skipping of sections of code using 'saved' and 'regen'.
    
    :arg approach: user specified input.
    :return: approach to use and keys to skip sections.
    """
    approach = approach or default
    if approach in ('norm', 'fieldBased', 'gradientBased', 'hessianBased', 'fluxJump'):
        getData = False
        getError = False
        useAdjoint = False
        aposteriori = False
    elif approach in ('residual', 'explicit', 'implicit', 'DWR', 'DWE', 'DWF'):
        getData = True
        getError = True
        useAdjoint = approach in ('DWR', 'DWE', 'DWF')
        aposteriori = True
    elif approach in ('saved', 'regen'):
        getError = approach == 'regen'
        approach = input("""Choose error estimator from 'residual', 'explicit', 'implicit', 'DWF', 'DWR' or 'DWE': """)\
                   or 'DWR'
        getData = False
        useAdjoint = approach in ('DWR', 'DWE', 'DWF')
        aposteriori = True
    else:
        approach = 'fixedMesh'
        getData = True
        getError = False
        useAdjoint = False
        aposteriori = False

    return approach, getData, getError, useAdjoint, aposteriori


def printTimings(primal, dual=False, error=False, adapt=False, full=False):
    """
    Print timings for the various sections of code.
    
    :arg primal: primal solver.
    :arg dual: dual solver.
    :arg error: error estimation phase.
    :arg adapt: adaptive primal solver.
    :arg full: total time to solution.
    """
    print("TIMINGS:")
    if bool(primal):
        print("Forward run   %5.3fs" % primal)
    if bool(dual):
        print("Adjoint run   %5.3fs" % dual)
    if bool(error):
        print("Error run     %5.3fs" % error)
    if bool(adapt):
        print("Adaptive run  %5.3fs" % adapt)
    if bool(full):
        print("Setups        %5.3fs" % (float(full) - (primal + float(dual) + float(error) + float(adapt))))
        print("Total         %5.3fs\n" % full)


def getMax(array):
    """
    :param array: 1D array.
    :return: index for maximum and its value. 
    """
    i = 0
    m = 0
    for j in range(len(array)):
        if array[j] > m:
            m = array[j]
            i = j
    return i, m
