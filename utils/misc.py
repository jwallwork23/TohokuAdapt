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
    if approach in ('norm', 'fieldBased', 'gradientBased', 'hessianBased'):
        getData = False
        getError = False
        useAdjoint = False
    elif approach in ('residual', 'explicit', 'fluxJump', 'implicit', 'implicitNorm'):
        getData = True
        getError = True
        useAdjoint = False
    elif approach in ('DWR', 'DWE', 'DWF'):
        getData = True
        getError = True
        useAdjoint = True
    elif approach == 'saved':
        approach = \
            input("""Choose error estimator from 
    'residual', 'explicit', 'fluxJump', 'implicit', 'implicitNorm', 'DWF', 'DWR' or 'DWE': """) \
            or 'DWR'
        getData = False
        getError = False
        useAdjoint = True
    elif approach == 'regen':
        approach = input("""Choose error estimator from 
    'residual', 'explicit', 'fluxJump', 'implicit', 'implicitNorm', 'DWF', 'DWR' or 'DWE',: """) \
            or 'DWR'
        getData = False
        getError = True
        useAdjoint = approach in ('DWF', 'DWR')
    else:
        approach = 'fixedMesh'
        getData = True
        getError = False
        useAdjoint = False

    return approach, getData, getError, useAdjoint


def printTimings(primal, dual=False, error=False, adapt=False, boot=False):
    """
    Print timings for the various sections of code.
    
    :arg primal: primal solver.
    :arg dual: dual solver.
    :arg error: error estimation phase.
    :arg adapt: adaptive primal solver.
    :arg boot: bootstrapping routine.
    :return: 
    """
    print("TIMINGS:")
    if bool(boot):
        print("Bootstrap run %5.3fs" % boot)
    print("Forward run   %5.3fs" % primal)
    if bool(dual):
        print("Adjoint run   %5.3fs" % dual)
    if bool(error):
        print("Error run     %5.3fs" % error)
    if bool(adapt):
        print("Adaptive run  %5.3fs" % adapt)
    print("Total         %5.3fs\n" % (primal + float(dual) + float(error) + float(adapt) + float(boot)))


def dis(string, printStats=True):
    if printStats:
        print(string)
